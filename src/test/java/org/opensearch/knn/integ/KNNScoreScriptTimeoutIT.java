/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.KNNRestTestCase;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * {@code script_score} with {@code knn_score} on a real cluster:
 * <ul>
 *   <li>scores are unchanged by the timeout check, with a filter, a boost and {@code min_score};</li>
 *   <li>a search that times out mid-segment stops within about 1,000 docs, with or without {@code min_score},
 *       instead of at core's first scoring window boundary (4,096 docs);</li>
 *   <li>a search cancelled with {@code cancel_after_time_interval} stops about as soon.</li>
 * </ul>
 * The timeout tests use a slow Painless filter (about 1 ms per doc), so the deadline passes within the first few hundred
 * docs of the segment. With {@code track_total_hits}, the reported total is the number of docs scored before the
 * search stopped.
 */
public class KNNScoreScriptTimeoutIT extends KNNRestTestCase {

    private static final String INDEX = "knn-score-timeout";
    private static final String FIELD = "vec";
    private static final int DIMENSION = 4;
    private static final int NUM_DOCS = 10_000;
    // Core's first CancellableBulkScorer window. Without the fix, a timed-out search scores this many.
    private static final int CORE_FIRST_WINDOW = 4_096;
    private static final float[] QUERY = { 1.0f, 2.0f, 3.0f, 4.0f };
    // About 1 ms of Painless work per doc.
    private static final String SLOW_FILTER = "{\"script\":{\"script\":{\"source\":\"long s = 0; for (int i = 0; i < params.n; i++) "
        + "{ s += i; } return s >= 0;\",\"params\":{\"n\":300000}}}}";

    private static float[] vectorFor(int i) {
        return new float[] { i % 7, i % 11, i % 13, i % 17 };
    }

    private void createAndLoadIndex() throws Exception {
        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .endObject()
            .startObject("bucket")
            .field("type", "integer")
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(
            INDEX,
            Settings.builder()
                .put("index.number_of_shards", 1)
                .put("index.number_of_replicas", 0)
                .put("index.knn", true)
                .put("index.requests.cache.enable", false)
                .build(),
            mapping
        );
        StringBuilder bulk = new StringBuilder();
        for (int i = 0; i < NUM_DOCS; i++) {
            final float[] v = vectorFor(i);
            bulk.append(String.format(Locale.ROOT, "{\"index\":{\"_index\":\"%s\",\"_id\":\"%d\"}}%n", INDEX, i));
            bulk.append(String.format(Locale.ROOT, "{\"%s\":[%s,%s,%s,%s],\"bucket\":%d}%n", FIELD, v[0], v[1], v[2], v[3], i % 2));
            if ((i + 1) % 2_000 == 0) {
                sendBulk(bulk.toString());
                bulk = new StringBuilder();
            }
        }
        if (bulk.length() > 0) {
            sendBulk(bulk.toString());
        }
        forceMergeKnnIndex(INDEX, 1);
    }

    private void sendBulk(String body) throws Exception {
        final Request request = new Request("POST", "/_bulk");
        request.setJsonEntity(body);
        final Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        assertEquals(Boolean.FALSE, toMap(response).get("errors"));
    }

    private static String knnScoreScript() {
        return String.format(
            Locale.ROOT,
            "{\"source\":\"knn_score\",\"lang\":\"knn\",\"params\":{\"field\":\"%s\",\"query_value\":[%s,%s,%s,%s],\"space_type\":\"l2\"}}",
            FIELD,
            QUERY[0],
            QUERY[1],
            QUERY[2],
            QUERY[3]
        );
    }

    private static float expectedScore(int doc) {
        final float[] v = vectorFor(doc);
        float d = 0;
        for (int i = 0; i < DIMENSION; i++) {
            d += (v[i] - QUERY[i]) * (v[i] - QUERY[i]);
        }
        return 1 / (1 + d);
    }

    private Map<String, Object> toMap(Response response) throws Exception {
        return createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), EntityUtils.toString(response.getEntity())).map();
    }

    /** A slow-filtered knn_score search; size > 0 so the query needs scores and knn_score runs. */
    private static String slowKnnScoreSearch(String extra) {
        return String.format(
            Locale.ROOT,
            "{\"size\":10,\"track_total_hits\":true,%s\"query\":{\"script_score\":{\"query\":{\"bool\":{\"filter\":[%s]}},\"script\":%s%s}}}",
            extra,
            SLOW_FILTER,
            knnScoreScript(),
            ""
        );
    }

    @SuppressWarnings("unchecked")
    private static int totalHits(Map<String, Object> response) {
        return ((Number) ((Map<String, Object>) ((Map<String, Object>) response.get("hits")).get("total")).get("value")).intValue();
    }

    @SuppressWarnings("unchecked")
    public void testKnnScore_scoresMatchExpected_withFilterBoostAndMinScore() throws Exception {
        createAndLoadIndex();
        final List<Integer> odd = new ArrayList<>();
        for (int i = 1; i < NUM_DOCS; i += 2) {
            odd.add(i);
        }
        odd.sort(Comparator.comparingDouble((Integer i) -> -expectedScore(i)).thenComparingInt(i -> i));

        for (String minScore : new String[] { "", ",\"min_score\":0.0" }) {
            final int size = 20;
            final String body = String.format(
                Locale.ROOT,
                "{\"size\":%d,\"track_total_hits\":true,\"query\":{\"script_score\":{\"query\":{\"term\":{\"bucket\":1}},\"script\":%s,"
                    + "\"boost\":2.0%s}}}",
                size,
                knnScoreScript(),
                minScore
            );

            final Map<String, Object> response = toMap(performSearch(INDEX, body));

            assertEquals(Boolean.FALSE, response.get("timed_out"));
            assertEquals("every bucket 1 doc is scored", NUM_DOCS / 2, totalHits(response));
            final List<Map<String, Object>> hits = (List<Map<String, Object>>) ((Map<String, Object>) response.get("hits")).get("hits");
            assertEquals(size, hits.size());
            assertEquals(expectedScore(odd.get(0)) * 2.0f, ((Number) hits.get(0).get("_score")).floatValue(), 1e-5f);
            for (Map<String, Object> hit : hits) {
                final int id = Integer.parseInt((String) hit.get("_id"));
                assertEquals("only bucket 1 docs", 1, id % 2);
                assertEquals("score of doc " + id, expectedScore(id) * 2.0f, ((Number) hit.get("_score")).floatValue(), 1e-5f);
            }
        }
    }

    public void testKnnScore_whenTimeoutMidSegment_thenStopsWithinCheckInterval() throws Exception {
        createAndLoadIndex();

        final Map<String, Object> response = toMap(performSearch(INDEX, slowKnnScoreSearch("\"timeout\":\"10ms\",")));

        assertEquals(Boolean.TRUE, response.get("timed_out"));
        final int scored = totalHits(response);
        logger.info("knn_score scored [{}] of [{}] docs before the timeout stopped it", scored, NUM_DOCS);
        assertTrue(
            "scored " + scored + " docs; expected fewer than core's first window (" + CORE_FIRST_WINDOW + ")",
            scored < CORE_FIRST_WINDOW
        );
    }

    public void testKnnScore_whenTimeoutMidSegmentWithMinScore_thenStopsWithinCheckInterval() throws Exception {
        createAndLoadIndex();
        final String body = String.format(
            Locale.ROOT,
            "{\"size\":10,\"track_total_hits\":true,\"timeout\":\"10ms\",\"query\":{\"script_score\":{\"query\":{\"bool\":{\"filter\":[%s]}},"
                + "\"script\":%s,\"min_score\":0.0}}}",
            SLOW_FILTER,
            knnScoreScript()
        );

        final Map<String, Object> response = toMap(performSearch(INDEX, body));

        assertEquals(Boolean.TRUE, response.get("timed_out"));
        final int scored = totalHits(response);
        logger.info("knn_score with min_score scored [{}] of [{}] docs before the timeout stopped it", scored, NUM_DOCS);
        assertTrue(
            "scored " + scored + " docs; expected fewer than core's first window (" + CORE_FIRST_WINDOW + ")",
            scored < CORE_FIRST_WINDOW
        );
    }

    /**
     * {@code cancel_after_time_interval} cancels the search task, and the request fails as cancelled. What matters is
     * how long the shard keeps scoring after the cancel. The test first times how long the same slow query takes to
     * score docs (from a timed-out run, which also warms up the scripts), then checks the cancelled search stops within
     * about two check intervals of the cancel. Stopping at core's first window (4,096 docs) takes about four times longer.
     */
    public void testKnnScore_whenCancelAfterTimeInterval_thenStopsWithinCheckInterval() throws Exception {
        createAndLoadIndex();
        final Map<String, Object> timed = toMap(performSearch(INDEX, slowKnnScoreSearch("\"timeout\":\"500ms\",")));
        assertEquals(Boolean.TRUE, timed.get("timed_out"));
        final double msPerDoc = ((Number) timed.get("took")).doubleValue() / Math.max(1, totalHits(timed));

        final Request request = new Request("POST", "/" + INDEX + "/_search");
        request.addParameter("cancel_after_time_interval", "100ms");
        request.setJsonEntity(slowKnnScoreSearch(""));
        final long start = System.nanoTime();
        String outcome;
        try {
            outcome = EntityUtils.toString(client().performRequest(request).getEntity());
        } catch (ResponseException e) {
            outcome = EntityUtils.toString(e.getResponse().getEntity());
        }
        final long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        final double docsAfterCancel = Math.max(0, elapsedMs - 100) / msPerDoc;
        logger.info(
            "cancelled search returned after [{}] ms at [{}] ms/doc: about [{}] docs scored after the cancel",
            elapsedMs,
            String.format(Locale.ROOT, "%.3f", msPerDoc),
            Math.round(docsAfterCancel)
        );

        assertTrue("expected a cancelled search, got " + outcome, outcome.contains("task_cancelled_exception"));
        assertTrue(
            "about "
                + Math.round(docsAfterCancel)
                + " docs scored after the cancel; expected under 2 check intervals, well below "
                + CORE_FIRST_WINDOW,
            docsAfterCancel < 2 * 1000
        );
    }
}
