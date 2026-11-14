/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreMode;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

/*
 * A factory that creates KNNScoreScriptLeafFactory objects. The factory is responsible for parsing the parameters
 * passed in the query and creating the KNNScoreScriptLeafFactory object.
 */
public class KNNScoreScriptLeafFactory implements ScoreScript.BulkScoringLeafFactory {
    private final Map<String, Object> params;
    private final SearchLookup lookup;
    private final String similaritySpace;
    private final String field;
    private final Object query;
    private final KNNScoringSpace knnScoringSpace;
    private final IndexSearcher searcher;

    public KNNScoreScriptLeafFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher) {
        KNNCounter.SCRIPT_QUERY_REQUESTS.increment();
        this.params = params;
        this.lookup = lookup;
        this.field = getValue(params, "field").toString();
        this.similaritySpace = getValue(params, "space_type").toString();
        this.query = getValue(params, "query_value");
        this.searcher = searcher;

        this.knnScoringSpace = KNNScoringSpaceFactory.create(
            this.similaritySpace,
            this.query,
            lookup.doc().mapperService().fieldType(this.field)
        );
    }

    private Object getValue(Map<String, Object> params, String fieldName) {
        final Object value = params.get(fieldName);
        if (value != null) return value;

        KNNCounter.SCRIPT_QUERY_ERRORS.increment();
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Missing parameter [%s]", fieldName));
    }

    @Override
    public boolean needs_score() {
        return false;
    }

    /**
     * For each segment, supply the KNNScoreScript that should be used to re-score the documents returned from the
     * query. Because the method to score the documents was set during factory construction, the scripts are agnostic of
     * the similarity space. The KNNScoringSpace will return the correct script, given the query, the field type, and
     * the similarity space.
     *
     * @param ctx LeafReaderContext for the segment
     * @return ScoreScript to be executed
     */
    @Override
    public ScoreScript newInstance(LeafReaderContext ctx) throws IOException {
        return knnScoringSpace.getScoreScript(params, field, lookup, ctx, this.searcher);
    }

    /**
     * Scores {@code script_score} in doc ID spans so the search's timeout and task cancellation are checked every
     * {@link KNNScoreBulkScorer#DOCS_BETWEEN_TIMEOUT_CHECK} docs, instead of once per core scoring window (4,096 doc IDs,
     * doubling up to 1,048,576). Core only calls this when {@code min_score} isn't set.
     * <p>
     * Returns null, so core keeps its default per-document scorer, when the searcher has no timeout to check. The
     * script's index name, shard and index version aren't set here (core sets them on its default path); {@code knn_score}
     * scripts don't read them.
     */
    @Override
    public BulkScorer bulkScorer(
        final LeafReaderContext context,
        final BulkScorer subQueryBulkScorer,
        final ScoreMode subQueryScoreMode,
        final float boost
    ) throws IOException {
        return KNNScoreBulkScorer.create(this, searcher, context, subQueryBulkScorer, subQueryScoreMode, boost);
    }
}
