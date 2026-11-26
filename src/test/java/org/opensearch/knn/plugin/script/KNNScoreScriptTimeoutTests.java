/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.lookup.LeafSearchLookup;
import org.opensearch.search.lookup.SearchLookup;
import org.opensearch.search.query.QueryPhase;

import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * {@code knn_score} runs core's timeout and task cancellation check every {@link KNNScoreScript#DOCS_BETWEEN_TIMEOUT_CHECK}
 * docs, and lets core's own exception through, instead of relying only on core's per-window check.
 */
public class KNNScoreScriptTimeoutTests extends KNNTestCase {

    private static final int INTERVAL = KNNScoreScript.DOCS_BETWEEN_TIMEOUT_CHECK;

    private LeafSearchLookup leafLookup;

    /**
     * Shaped like core 3.7's {@code ContextIndexSearcher.MutableQueryTimeout}: a private class whose public
     * {@code checkCancelled()} throws the exception of whichever check fired.
     */
    private static class CoreLikeTimeout implements QueryTimeout {
        private final AtomicInteger calls = new AtomicInteger();
        private final IntConsumer onCheck;

        private CoreLikeTimeout(IntConsumer onCheck) {
            this.onCheck = onCheck;
        }

        public void checkCancelled() {
            onCheck.accept(calls.incrementAndGet());
        }

        @Override
        public boolean shouldExit() {
            throw new AssertionError("knn_score must run checkCancelled(), which throws the right exception");
        }
    }

    private KNNScoreScript<Long> script(final IndexSearcher searcher) {
        final SearchLookup lookup = mock(SearchLookup.class);
        leafLookup = mock(LeafSearchLookup.class);
        when(lookup.getLeafSearchLookup(any())).thenReturn(leafLookup);
        when(leafLookup.asMap()).thenReturn(Collections.emptyMap());
        return new KNNScoreScript.LongType(
            Collections.emptyMap(),
            1L,
            "field",
            (a, b) -> 1.0f,
            lookup,
            mock(LeafReaderContext.class),
            searcher
        );
    }

    private static IndexSearcher searcherWithTimeout(final QueryTimeout timeout) {
        final IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.getTimeout()).thenReturn(timeout);
        return searcher;
    }

    /** Scores docs until the script throws; returns how many were scored. */
    private int scoreUntilStopped(final KNNScoreScript<Long> script, final int maxDocs, final Class<? extends RuntimeException> expected) {
        int scored = 0;
        try {
            for (int doc = 0; doc < maxDocs; doc++) {
                script.setDocument(doc);
                scored++;
            }
            fail("expected the script to stop the search");
        } catch (RuntimeException e) {
            assertSame("core's exception must pass through unchanged", expected, e.getClass());
        }
        return scored;
    }

    public void testSetDocument_whenNothingFires_thenChecksOncePerIntervalAndScoresEveryDoc() {
        final CoreLikeTimeout timeout = new CoreLikeTimeout(call -> {});
        final KNNScoreScript<Long> script = script(searcherWithTimeout(timeout));

        final int docs = 10 * INTERVAL + 500;
        for (int doc = 0; doc < docs; doc++) {
            script.setDocument(doc);
        }

        assertEquals(10, timeout.calls.get());
        verify(leafLookup, times(docs)).setDocument(anyInt());
    }

    public void testSetDocument_whenTimeoutFires_thenThrowsCoresTimeExceededWithinOneInterval() {
        final CoreLikeTimeout timeout = new CoreLikeTimeout(call -> {
            if (call >= 3) {
                throw new QueryPhase.TimeExceededException();
            }
        });

        final int scored = scoreUntilStopped(script(searcherWithTimeout(timeout)), 100 * INTERVAL, QueryPhase.TimeExceededException.class);

        assertEquals(3 * INTERVAL - 1, scored);
        verify(leafLookup, times(scored)).setDocument(anyInt());
    }

    public void testSetDocument_whenTaskCancelled_thenThrowsTaskCancelledNotTimeout() {
        final CoreLikeTimeout timeout = new CoreLikeTimeout(call -> {
            if (call >= 2) {
                throw new TaskCancelledException("cancelled task with reason: test");
            }
        });

        final int scored = scoreUntilStopped(script(searcherWithTimeout(timeout)), 100 * INTERVAL, TaskCancelledException.class);

        assertEquals(2 * INTERVAL - 1, scored);
    }

    public void testSetDocument_whenAlreadyFired_thenThrowsAtFirstCheck() {
        final KNNScoreScript<Long> script = script(
            searcherWithTimeout(new CoreLikeTimeout(call -> { throw new QueryPhase.TimeExceededException(); }))
        );

        for (int doc = 0; doc < INTERVAL - 1; doc++) {
            script.setDocument(doc);
        }
        expectThrows(QueryPhase.TimeExceededException.class, () -> script.setDocument(INTERVAL - 1));
    }

    public void testSetDocument_whenNoSearcherNoTimeoutOrNoCoreCheck_thenNeverChecks() {
        final AtomicInteger shouldExitCalls = new AtomicInteger();
        // A plain QueryTimeout has no checkCancelled(): knn_score can't tell a timeout from a cancellation, so it adds
        // no check and leaves it to core.
        final QueryTimeout plain = () -> {
            shouldExitCalls.incrementAndGet();
            return true;
        };
        for (final IndexSearcher searcher : new IndexSearcher[] { null, searcherWithTimeout(null), searcherWithTimeout(plain) }) {
            final KNNScoreScript<Long> script = script(searcher);
            for (int doc = 0; doc < 5 * INTERVAL; doc++) {
                script.setDocument(doc);
            }
            verify(leafLookup, times(5 * INTERVAL)).setDocument(anyInt());
        }
        assertEquals(0, shouldExitCalls.get());
    }

    public void testCancellationCheck_isResolvedFromCoreShapedTimeout() {
        assertNotNull(KNNScoreScript.cancellationCheck(searcherWithTimeout(new CoreLikeTimeout(call -> {}))));
        assertNull(KNNScoreScript.cancellationCheck(searcherWithTimeout(() -> false)));
        assertNull(KNNScoreScript.cancellationCheck(null));
    }

    /** Against core 3.7's real ContextIndexSearcher: the check runs core's registered checks and keeps their exception. */
    public void testCancellationCheck_withRealContextIndexSearcher_thenThrowsCoresExceptions() throws Exception {
        try (Directory directory = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
                writer.addDocument(new Document());
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final ContextIndexSearcher searcher = new ContextIndexSearcher(
                    reader,
                    new BM25Similarity(),
                    IndexSearcher.getDefaultQueryCache(),
                    IndexSearcher.getDefaultQueryCachingPolicy(),
                    true,
                    null,
                    mock(SearchContext.class)
                );
                // No checks registered yet (no timeout, low-level cancellation off): nothing fires.
                final Runnable check = KNNScoreScript.cancellationCheck(searcher);
                assertNotNull("core 3.7's searcher timeout must expose checkCancelled()", check);
                check.run();

                final Runnable timeout = searcher.addQueryCancellation(() -> { throw new QueryPhase.TimeExceededException(); });
                expectThrows(QueryPhase.TimeExceededException.class, check::run);
                searcher.removeQueryCancellation(timeout);

                searcher.addQueryCancellation(() -> { throw new TaskCancelledException("cancelled task with reason: test"); });
                expectThrows(TaskCancelledException.class, check::run);
            }
        }
    }

    public void testCheckInterval_isOneThousandDocs() {
        // Core's first scoring window is 4,096 docs; the check must run well inside it.
        assertEquals(1000, INTERVAL);
    }
}
