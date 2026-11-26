/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.IndexSearcher;
import org.opensearch.knn.index.KNNVectorScriptDocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.math.BigInteger;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * KNNScoreScript is used for adjusting the score of query results based on similarity distance methods. Scripts
 * operate on a per document basis. Because the scoring method is passed in during construction, KNNScoreScripts are
 * only concerned with the types of the query and docs being processed.
 */
@Log4j2
public abstract class KNNScoreScript<T> extends ScoreScript {
    /**
     * How often to check the search's timeout and task cancellation while scoring. Core only checks once per scoring
     * window (4,096 docs, doubling up to 1,048,576), so without this a cancelled search keeps scoring for up to a full
     * window. Matches Lucene's {@code ExitableDirectoryReader} interval.
     */
    static final int DOCS_BETWEEN_TIMEOUT_CHECK = 1000;

    protected final T queryValue;
    protected final String field;
    protected final BiFunction<T, T, Float> scoringMethod;
    // Runs core's own timeout and cancellation check, which throws the same exception core's checks throw:
    // TimeExceededException (core keeps the results so far and reports timed_out) or TaskCancelledException. Null when
    // the script doesn't run inside a search with these checks, in which case nothing is checked.
    private final Runnable cancellationCheck;
    private int docsSinceTimeoutCheck;

    public KNNScoreScript(
        Map<String, Object> params,
        T queryValue,
        String field,
        BiFunction<T, T, Float> scoringMethod,
        SearchLookup lookup,
        LeafReaderContext leafContext,
        IndexSearcher searcher
    ) {
        super(params, lookup, searcher, leafContext);
        this.queryValue = queryValue;
        this.field = field;
        this.scoringMethod = scoringMethod;
        this.cancellationCheck = cancellationCheck(searcher);
    }

    /**
     * Runs core's timeout and task cancellation check every {@link #DOCS_BETWEEN_TIMEOUT_CHECK} docs. Core catches a
     * timeout per segment and returns the results collected so far; a cancellation fails the search as cancelled. This
     * throws instead of returning fewer docs, so a cut-short result can't end up in a cache.
     */
    @Override
    public void setDocument(int docId) {
        if (cancellationCheck != null && ++docsSinceTimeoutCheck >= DOCS_BETWEEN_TIMEOUT_CHECK) {
            docsSinceTimeoutCheck = 0;
            cancellationCheck.run();
        }
        super.setDocument(docId);
    }

    /**
     * Core 3.7 sets the search's timeout on the searcher as an object whose {@code checkCancelled()} runs every
     * registered timeout and cancellation check and throws the matching exception. It isn't public API, so it's found
     * reflectively once per segment; when it isn't there, no check is added and scoring behaves as before.
     */
    static Runnable cancellationCheck(final IndexSearcher searcher) {
        final QueryTimeout timeout = searcher == null ? null : searcher.getTimeout();
        if (timeout == null) {
            return null;
        }
        final Method check;
        try {
            check = timeout.getClass().getMethod("checkCancelled");
            check.setAccessible(true);
        } catch (NoSuchMethodException | RuntimeException e) {
            log.debug("[KNN] knn_score can't run the search's cancellation check; relying on core's checks", e);
            return null;
        }
        return () -> {
            try {
                check.invoke(timeout);
            } catch (InvocationTargetException e) {
                final Throwable cause = e.getCause();
                if (cause instanceof RuntimeException) {
                    throw (RuntimeException) cause;
                }
                if (cause instanceof Error) {
                    throw (Error) cause;
                }
                throw new IllegalStateException(cause);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        };
    }

    /**
     * KNNScoreScript with Long type. The query value passed in as well as the DocValues being searched over are
     * expected to be Longs.
     */
    public static class LongType extends KNNScoreScript<Long> {
        public LongType(
            Map<String, Object> params,
            Long queryValue,
            String field,
            BiFunction<Long, Long, Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext,
            IndexSearcher searcher
        ) {
            super(params, queryValue, field, scoringMethod, lookup, leafContext, searcher);
        }

        /**
         * This function calculates the similarity score for each doc in the segment.
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score for the provided space between the doc and the query
         */
        @Override
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            ScriptDocValues.Longs scriptDocValues = (ScriptDocValues.Longs) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, scriptDocValues.getValue());
        }
    }

    /**
     * KNNScoreScript with BigInteger type. The query value passed in as well as the DocValues being searched over
     * are expected to be BigInteger.
     */
    public static class BigIntegerType extends KNNScoreScript<BigInteger> {
        public BigIntegerType(
            Map<String, Object> params,
            BigInteger queryValue,
            String field,
            BiFunction<BigInteger, BigInteger, Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext,
            IndexSearcher searcher
        ) {
            super(params, queryValue, field, scoringMethod, lookup, leafContext, searcher);
        }

        /**
         * This function calculates the similarity score for each doc in the segment.
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score for the provided space between the doc and the query
         */
        @Override
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            ScriptDocValues.BytesRefs scriptDocValues = (ScriptDocValues.BytesRefs) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, new BigInteger(1, scriptDocValues.getValue().bytes));
        }
    }

    /**
     * KNNVectors with float[] type. The query value passed in is expected to be float[]. The fieldType of the docs
     * being searched over are expected to be KNNVector type.
     */
    public static class KNNFloatVectorType extends KNNScoreScript<float[]> {

        public KNNFloatVectorType(
            Map<String, Object> params,
            float[] queryValue,
            String field,
            BiFunction<float[], float[], Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext,
            IndexSearcher searcher
        ) throws IOException {
            super(params, queryValue, field, scoringMethod, lookup, leafContext, searcher);
        }

        /**
         * This function called for each doc in the segment. We evaluate the score of the vector in the doc
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score of the vector to the query vector
         */
        @Override
        @SuppressWarnings("unchecked")
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            KNNVectorScriptDocValues<float[]> scriptDocValues = (KNNVectorScriptDocValues<float[]>) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, scriptDocValues.getValue());
        }
    }

    /**
     * KNNVectors with byte[] type. The query value passed in is expected to be byte[]. The fieldType of the docs
     * being searched over are expected to be KNNVector type.
     */
    public static class KNNByteVectorType extends KNNScoreScript<byte[]> {

        public KNNByteVectorType(
            Map<String, Object> params,
            byte[] queryValue,
            String field,
            BiFunction<byte[], byte[], Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext,
            IndexSearcher searcher
        ) throws IOException {
            super(params, queryValue, field, scoringMethod, lookup, leafContext, searcher);
        }

        /**
         * This function called for each doc in the segment. We evaluate the score of the vector in the doc
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score of the vector to the query vector
         */
        @Override
        @SuppressWarnings("unchecked")
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            KNNVectorScriptDocValues<byte[]> scriptDocValues = (KNNVectorScriptDocValues<byte[]>) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, scriptDocValues.getValue());
        }
    }
}
