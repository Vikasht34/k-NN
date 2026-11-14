/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.FilterLeafCollector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.util.Bits;
import org.opensearch.script.ScoreScript;

import java.io.IOException;
import java.util.Objects;

/**
 * {@link BulkScorer} for {@code script_score} queries that use {@code knn_score}. It scores documents exactly like core's
 * default {@code ScriptScoreQuery} bulk scorer (same script, same score validation, same boost), but drives the
 * sub-query in spans of {@link #DOCS_BETWEEN_TIMEOUT_CHECK} doc IDs and checks the search's {@link QueryTimeout} between
 * spans.
 * <p>
 * Without this, the only check inside a segment is core's {@code CancellableBulkScorer}, which checks before each
 * scoring window of 4,096 doc IDs that doubles up to 1,048,576. A timed-out or cancelled query then keeps scoring up to a
 * million vectors per check.
 * <p>
 * When the timeout fires (request timeout or task cancellation), {@link #score} returns the first doc ID it hasn't scored.
 * Every doc before it was scored once and none after it was. Core's {@code CancellableBulkScorer}, which wraps every
 * top-level bulk scorer while cancellation is enabled, then runs its cancellation check, which throws the timeout
 * (partial results, {@code timed_out: true}) or the task-cancelled exception. If nothing throws, the caller continues from
 * the returned doc, so no document is dropped or scored twice.
 */
final class KNNScoreBulkScorer extends BulkScorer {

    /** Doc ID span between timeout checks. Matches Lucene's {@code ExitableDirectoryReader}. */
    static final int DOCS_BETWEEN_TIMEOUT_CHECK = 1000;

    private final BulkScorer subQueryBulkScorer;
    private final ScoreMode subQueryScoreMode;
    private final ScoreScript scoreScript;
    private final float boost;
    private final QueryTimeout queryTimeout;

    KNNScoreBulkScorer(
        final BulkScorer subQueryBulkScorer,
        final ScoreMode subQueryScoreMode,
        final ScoreScript scoreScript,
        final float boost,
        final QueryTimeout queryTimeout
    ) {
        this.subQueryBulkScorer = Objects.requireNonNull(subQueryBulkScorer);
        this.subQueryScoreMode = subQueryScoreMode;
        this.scoreScript = Objects.requireNonNull(scoreScript);
        this.boost = boost;
        this.queryTimeout = Objects.requireNonNull(queryTimeout);
    }

    /**
     * The bulk scorer for {@code ScriptScoreQuery} to use, or null to keep core's default per-document scorer when the
     * searcher has no timeout to check.
     *
     * @param scripts            creates the per-segment score script
     * @param searcher           the searcher running the query; its {@link IndexSearcher#getTimeout()} is checked
     * @param context            the segment
     * @param subQueryBulkScorer drives document iteration
     * @param subQueryScoreMode  score mode of the sub-query
     * @param boost              boost applied to every score
     */
    static BulkScorer create(
        final ScoreScript.LeafFactory scripts,
        final IndexSearcher searcher,
        final LeafReaderContext context,
        final BulkScorer subQueryBulkScorer,
        final ScoreMode subQueryScoreMode,
        final float boost
    ) throws IOException {
        final QueryTimeout queryTimeout = searcher == null ? null : searcher.getTimeout();
        if (queryTimeout == null) {
            return null;
        }
        return new KNNScoreBulkScorer(subQueryBulkScorer, subQueryScoreMode, scripts.newInstance(context), boost, queryTimeout);
    }

    @Override
    public int score(final LeafCollector collector, final Bits acceptDocs, final int min, final int max) throws IOException {
        final LeafCollector scoringCollector = new ScoringCollector(collector);
        int doc = min;
        boolean firstSpan = true;
        while (doc < max) {
            // The caller checks before handing over a window, so the first span doesn't need its own check.
            if (firstSpan == false && queryTimeout.shouldExit()) {
                return doc;
            }
            firstSpan = false;
            final int spanEnd = (int) Math.min((long) doc + DOCS_BETWEEN_TIMEOUT_CHECK, max);
            doc = subQueryBulkScorer.score(scoringCollector, acceptDocs, doc, spanEnd);
        }
        return doc;
    }

    @Override
    public long cost() {
        return subQueryBulkScorer.cost();
    }

    /** Gives the downstream collector a {@link Scorable} that returns the script's score for the current doc. */
    private final class ScoringCollector extends FilterLeafCollector {
        private ScriptScorable scriptScorable;

        ScoringCollector(final LeafCollector in) {
            super(in);
        }

        @Override
        public void setScorer(final Scorable subQueryScorer) throws IOException {
            scriptScorable = new ScriptScorable(subQueryScorer);
            in.setScorer(scriptScorable);
        }

        @Override
        public void collect(final int doc) throws IOException {
            scriptScorable.docId = doc;
            super.collect(doc);
        }
    }

    /** Same scoring and validation as core's {@code ScriptScoreQuery.ScriptScorable}. */
    private final class ScriptScorable extends Scorable {
        private int docId;

        ScriptScorable(final Scorable subQueryScorer) {
            if (subQueryScoreMode == ScoreMode.COMPLETE) {
                scoreScript.setScorer(subQueryScorer);
            }
        }

        @Override
        public float score() throws IOException {
            scoreScript.setDocument(docId);
            final float score = (float) scoreScript.execute(null);
            if (score < 0f || Float.isNaN(score)) {
                throw new IllegalArgumentException(
                    "script_score script returned an invalid score [" + score + "] for doc [" + docId + "]. Must be a non-negative score!"
                );
            }
            return score * boost;
        }
    }
}
