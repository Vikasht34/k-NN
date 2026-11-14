/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.opensearch.Version;
import org.opensearch.common.lucene.search.function.ScriptScoreQuery;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.index.shard.SearchOperationListener;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.Script;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.lookup.DocLookup;
import org.opensearch.search.lookup.LeafSearchLookup;
import org.opensearch.search.lookup.SearchLookup;
import org.opensearch.search.query.QueryPhase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntToDoubleFunction;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.plugin.script.KNNScoreBulkScorer.DOCS_BETWEEN_TIMEOUT_CHECK;

/**
 * Timeout and task cancellation for {@code script_score} with {@code knn_score}, scored through
 * {@link KNNScoreBulkScorer}.
 */
public class KNNScoreBulkScorerTests extends KNNTestCase {

    // More than core's first CancellableBulkScorer window (4,096 doc IDs), so the default path's coarser check shows.
    private static final int NUM_DOCS = 5_000;
    // Scored docs after which the test timeout / cancellation fires.
    private static final int FIRE_AFTER = 1_500;
    private static final float BOOST = 2.0f;

    private Directory directory;
    private DirectoryReader reader;
    private LeafReaderContext leafContext;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document document = new Document();
                document.add(new StringField("id", Integer.toString(i), Field.Store.NO));
                writer.addDocument(document);
            }
            writer.forceMerge(1);
        }
        reader = DirectoryReader.open(directory);
        assertEquals(1, reader.leaves().size());
        leafContext = reader.leaves().get(0);
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        super.tearDown();
    }

    // ---- KNNScoreBulkScorer on its own ----

    public void testScore_whenNoTimeout_thenScoresEveryDocOnceWithBoost() throws IOException {
        final AtomicInteger executed = new AtomicInteger();
        final BulkScorer bulkScorer = new KNNScoreBulkScorer(
            matchAllBulkScorer(),
            ScoreMode.COMPLETE_NO_SCORES,
            script(leafContext, executed, doc -> doc + 1),
            BOOST,
            () -> false
        );
        final RecordingCollector collector = new RecordingCollector();

        final int next = bulkScorer.score(collector, null, 0, DocIdSetIterator.NO_MORE_DOCS);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, next);
        assertEquals(NUM_DOCS, collector.docs.size());
        for (int i = 0; i < NUM_DOCS; i++) {
            assertEquals(i, (int) collector.docs.get(i));
            assertEquals((i + 1) * BOOST, collector.scores.get(i), 0f);
        }
    }

    public void testScore_whenTimeoutFires_thenReturnsNextUnscoredDocWithinOneSpan_andResumesWithoutLossOrDuplicates() throws IOException {
        final AtomicInteger executed = new AtomicInteger();
        final AtomicBoolean allowTimeout = new AtomicBoolean(true);
        final QueryTimeout timeout = () -> allowTimeout.get() && executed.get() >= FIRE_AFTER;
        final BulkScorer bulkScorer = new KNNScoreBulkScorer(
            matchAllBulkScorer(),
            ScoreMode.COMPLETE_NO_SCORES,
            script(leafContext, executed, doc -> 1),
            1f,
            timeout
        );
        final RecordingCollector collector = new RecordingCollector();

        final int next = bulkScorer.score(collector, null, 0, DocIdSetIterator.NO_MORE_DOCS);

        assertTrue("stopped early, next=" + next, next < NUM_DOCS);
        assertTrue("scored " + executed.get() + " docs", executed.get() < FIRE_AFTER + DOCS_BETWEEN_TIMEOUT_CHECK);
        // Every doc before `next` was scored exactly once and none after it was.
        assertEquals(next, collector.docs.size());
        assertEquals(next - 1, (int) collector.docs.get(collector.docs.size() - 1));

        // If the caller keeps going from the returned doc, the rest of the segment is scored with nothing skipped.
        allowTimeout.set(false);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, bulkScorer.score(collector, null, next, DocIdSetIterator.NO_MORE_DOCS));
        assertEquals(NUM_DOCS, collector.docs.size());
        for (int i = 0; i < NUM_DOCS; i++) {
            assertEquals(i, (int) collector.docs.get(i));
        }
    }

    public void testScore_whenScriptReturnsInvalidScore_thenThrowsLikeCore() {
        for (double invalid : new double[] { -1.0, Double.NaN }) {
            final BulkScorer bulkScorer = new KNNScoreBulkScorer(
                matchAllBulkScorerUnchecked(),
                ScoreMode.COMPLETE_NO_SCORES,
                script(leafContext, new AtomicInteger(), doc -> invalid),
                1f,
                () -> false
            );
            final IllegalArgumentException e = expectThrows(
                IllegalArgumentException.class,
                () -> bulkScorer.score(new RecordingCollector(), null, 0, DocIdSetIterator.NO_MORE_DOCS)
            );
            assertTrue(e.getMessage(), e.getMessage().contains("Must be a non-negative score!"));
        }
    }

    public void testScore_whenSubQueryScoresNeeded_thenScriptGetsSubQueryScorer() throws IOException {
        for (ScoreMode mode : new ScoreMode[] { ScoreMode.COMPLETE, ScoreMode.COMPLETE_NO_SCORES }) {
            final AtomicBoolean scorerSet = new AtomicBoolean();
            final ScoreScript script = new ScoreScript(Collections.emptyMap(), lookup(), mock(IndexSearcher.class), leafContext) {
                @Override
                public void setScorer(Scorable scorer) {
                    scorerSet.set(true);
                }

                @Override
                public double execute(ExplanationHolder explanation) {
                    return 1.0;
                }
            };
            new KNNScoreBulkScorer(matchAllBulkScorer(), mode, script, 1f, () -> false).score(
                new RecordingCollector(),
                null,
                0,
                DocIdSetIterator.NO_MORE_DOCS
            );
            assertEquals(mode.toString(), mode == ScoreMode.COMPLETE, scorerSet.get());
        }
    }

    // ---- End to end: core's ContextIndexSearcher and ScriptScoreQuery ----

    public void testScriptScore_whenTaskCancelledMidSegment_thenThrowsWithinOneSpan() throws IOException {
        final AtomicInteger executed = new AtomicInteger();
        final SearchContext searchContext = searchContext();
        final ContextIndexSearcher searcher = contextSearcher(searchContext);
        searcher.addQueryCancellation(() -> {
            if (executed.get() >= FIRE_AFTER) {
                throw new TaskCancelledException("cancelled");
            }
        });

        expectThrows(
            TaskCancelledException.class,
            () -> searcher.search(scriptScore(knnStyleFactory(searcher, executed)), new ScoringCollector())
        );

        assertTrue("scored " + executed.get() + " docs", executed.get() < FIRE_AFTER + DOCS_BETWEEN_TIMEOUT_CHECK);
    }

    public void testScriptScore_whenTimeoutMidSegment_thenTimesOutWithinOneSpan() throws IOException {
        final AtomicInteger executed = new AtomicInteger();
        final SearchContext searchContext = searchContext();
        final ContextIndexSearcher searcher = contextSearcher(searchContext);
        searcher.addQueryCancellation(() -> {
            if (executed.get() >= FIRE_AFTER) {
                throw new QueryPhase.TimeExceededException();
            }
        });

        searcher.search(scriptScore(knnStyleFactory(searcher, executed)), new ScoringCollector());

        verify(searchContext).setSearchTimedOut(true);
        assertTrue("scored " + executed.get() + " docs", executed.get() < FIRE_AFTER + DOCS_BETWEEN_TIMEOUT_CHECK);
    }

    public void testScriptScore_withoutBulkScorer_thenCancellationOnlyAtCoreWindowBoundary() throws IOException {
        // Baseline for the two tests above: core's default per-document scorer only checks once its first 4,096 doc
        // window is done, so it scores far more docs after the cancel than the knn_score bulk scorer does.
        final AtomicInteger executed = new AtomicInteger();
        final ContextIndexSearcher searcher = contextSearcher(searchContext());
        searcher.addQueryCancellation(() -> {
            if (executed.get() >= FIRE_AFTER) {
                throw new TaskCancelledException("cancelled");
            }
        });

        expectThrows(TaskCancelledException.class, () -> searcher.search(scriptScore(defaultFactory(executed)), new ScoringCollector()));

        assertTrue("scored " + executed.get() + " docs", executed.get() >= FIRE_AFTER + DOCS_BETWEEN_TIMEOUT_CHECK);
    }

    public void testScriptScore_whenNotCancelled_thenSameScoresAsDefaultScorer() throws IOException {
        final IntToDoubleFunction scores = doc -> (doc % 97) + 0.5;

        final SearchContext bulkContext = searchContext();
        final ContextIndexSearcher bulkSearcher = contextSearcher(bulkContext);
        bulkSearcher.addQueryCancellation(() -> {});
        final ScoringCollector bulk = new ScoringCollector();
        bulkSearcher.search(scriptScore(knnStyleFactory(bulkSearcher, new AtomicInteger(), scores)), bulk);

        final ScoringCollector perDoc = new ScoringCollector();
        contextSearcher(searchContext()).search(scriptScore(defaultFactory(new AtomicInteger(), scores)), perDoc);

        assertEquals(NUM_DOCS, bulk.scores.size());
        assertEquals(perDoc.scores, bulk.scores);
        verify(bulkContext, never()).setSearchTimedOut(true);
    }

    // ---- KNNScoreScriptLeafFactory wiring ----

    public void testLeafFactory_isBulkScoringLeafFactory_andFallsBackWithoutTimeout() throws IOException {
        final IndexSearcher noTimeout = mock(IndexSearcher.class);
        final KNNScoreScriptLeafFactory withoutTimeout = new KNNScoreScriptLeafFactory(knnScoreParams(), knnLookup(), noTimeout);
        assertTrue(withoutTimeout instanceof ScoreScript.BulkScoringLeafFactory);
        assertNull(withoutTimeout.bulkScorer(leafContext, matchAllBulkScorer(), ScoreMode.COMPLETE_NO_SCORES, 1f));

        final IndexSearcher withTimeoutSearcher = mock(IndexSearcher.class);
        when(withTimeoutSearcher.getTimeout()).thenReturn(() -> false);
        final KNNScoreScriptLeafFactory withTimeout = new KNNScoreScriptLeafFactory(knnScoreParams(), knnLookup(), withTimeoutSearcher);
        assertTrue(
            withTimeout.bulkScorer(leafContext, matchAllBulkScorer(), ScoreMode.COMPLETE_NO_SCORES, 1f) instanceof KNNScoreBulkScorer
        );
    }

    // ---- helpers ----

    private BulkScorer matchAllBulkScorer() throws IOException {
        final IndexSearcher searcher = new IndexSearcher(reader);
        final Weight weight = searcher.createWeight(searcher.rewrite(new MatchAllDocsQuery()), ScoreMode.COMPLETE_NO_SCORES, 1f);
        return weight.scorerSupplier(leafContext).bulkScorer();
    }

    private BulkScorer matchAllBulkScorerUnchecked() {
        try {
            return matchAllBulkScorer();
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    private static SearchLookup lookup() {
        final SearchLookup lookup = mock(SearchLookup.class);
        when(lookup.getLeafSearchLookup(any())).thenReturn(mock(LeafSearchLookup.class));
        return lookup;
    }

    /** A score script that counts executions and returns {@code scoreForDoc} of the current doc. */
    private static ScoreScript script(final LeafReaderContext ctx, final AtomicInteger executed, final IntToDoubleFunction scoreForDoc) {
        return new ScoreScript(Collections.emptyMap(), lookup(), mock(IndexSearcher.class), ctx) {
            @Override
            public double execute(ExplanationHolder explanation) {
                executed.incrementAndGet();
                return scoreForDoc.applyAsDouble(_getDocId());
            }
        };
    }

    /** Same bulk-scorer decision as {@link KNNScoreScriptLeafFactory}, with a test script. */
    private static ScoreScript.BulkScoringLeafFactory knnStyleFactory(final IndexSearcher searcher, final AtomicInteger executed) {
        return knnStyleFactory(searcher, executed, doc -> 1.0);
    }

    private static ScoreScript.BulkScoringLeafFactory knnStyleFactory(
        final IndexSearcher searcher,
        final AtomicInteger executed,
        final IntToDoubleFunction scores
    ) {
        return new ScoreScript.BulkScoringLeafFactory() {
            @Override
            public boolean needs_score() {
                return false;
            }

            @Override
            public ScoreScript newInstance(LeafReaderContext ctx) {
                return script(ctx, executed, scores);
            }

            @Override
            public BulkScorer bulkScorer(LeafReaderContext ctx, BulkScorer subQueryBulkScorer, ScoreMode mode, float boost)
                throws IOException {
                return KNNScoreBulkScorer.create(this, searcher, ctx, subQueryBulkScorer, mode, boost);
            }
        };
    }

    /** A plain leaf factory, so core uses its default per-document scorer. */
    private static ScoreScript.LeafFactory defaultFactory(final AtomicInteger executed) {
        return defaultFactory(executed, doc -> 1.0);
    }

    private static ScoreScript.LeafFactory defaultFactory(final AtomicInteger executed, final IntToDoubleFunction scores) {
        return new ScoreScript.LeafFactory() {
            @Override
            public boolean needs_score() {
                return false;
            }

            @Override
            public ScoreScript newInstance(LeafReaderContext ctx) {
                return script(ctx, executed, scores);
            }
        };
    }

    private static ScriptScoreQuery scriptScore(final ScoreScript.LeafFactory factory) {
        return new ScriptScoreQuery(new MatchAllDocsQuery(), new Script("knn_score"), factory, null, "index", 0, Version.CURRENT);
    }

    private static SearchContext searchContext() {
        final SearchContext searchContext = mock(SearchContext.class);
        when(searchContext.bucketCollectorProcessor()).thenReturn(SearchContext.NO_OP_BUCKET_COLLECTOR_PROCESSOR);
        // ContextIndexSearcher notifies the shard's search listener; same setup as core's SearchCancellationTests.
        final IndexShard indexShard = mock(IndexShard.class);
        when(searchContext.indexShard()).thenReturn(indexShard);
        when(indexShard.getSearchOperationListener()).thenReturn(new SearchOperationListener() {
        });
        return searchContext;
    }

    private ContextIndexSearcher contextSearcher(final SearchContext searchContext) throws IOException {
        // No exitable reader, so these tests only see the bulk scorer's checks and core's CancellableBulkScorer.
        return new ContextIndexSearcher(
            reader,
            IndexSearcher.getDefaultSimilarity(),
            IndexSearcher.getDefaultQueryCache(),
            IndexSearcher.getDefaultQueryCachingPolicy(),
            false,
            null,
            searchContext
        );
    }

    private static Map<String, Object> knnScoreParams() {
        final Map<String, Object> params = new HashMap<>();
        params.put("field", "test");
        params.put("space_type", "l2");
        params.put("query_value", List.of(1.0, 2.0, 3.0));
        return params;
    }

    private static SearchLookup knnLookup() {
        final KNNVectorFieldType fieldType = new KNNVectorFieldType(
            "test",
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 3)
        );
        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType("test")).thenReturn(fieldType);
        final DocLookup docLookup = mock(DocLookup.class);
        when(docLookup.mapperService()).thenReturn(mapperService);
        final SearchLookup lookup = lookup();
        when(lookup.doc()).thenReturn(docLookup);
        return lookup;
    }

    /** Records each collected doc and its score, in order. */
    private static final class RecordingCollector implements LeafCollector {
        private final List<Integer> docs = new ArrayList<>();
        private final List<Float> scores = new ArrayList<>();
        private Scorable scorer;

        @Override
        public void setScorer(Scorable scorer) {
            this.scorer = scorer;
        }

        @Override
        public void collect(int doc) throws IOException {
            docs.add(doc);
            scores.add(scorer.score());
        }
    }

    /** Needs scores, so the query is scored (not wrapped in a constant score query) and every doc's score is read. */
    private static final class ScoringCollector extends SimpleCollector {
        private final List<Float> scores = new ArrayList<>();
        private Scorable scorer;

        @Override
        public void setScorer(Scorable scorer) {
            this.scorer = scorer;
        }

        @Override
        public void collect(int doc) throws IOException {
            scores.add(scorer.score());
        }

        @Override
        public ScoreMode scoreMode() {
            return ScoreMode.COMPLETE;
        }
    }
}
