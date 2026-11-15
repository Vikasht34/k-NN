/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.KNNRestTestCase;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * {@code script_score} with {@code knn_score} on a real cluster:
 * <ul>
 *   <li>scores are unchanged by the bulk scorer, with and without a filter and boost;</li>
 *   <li>when the search times out mid-segment, it stops within about 1,000 docs instead of at core's first scoring
 *       window boundary (4,096 docs).</li>
 * </ul>
 * The timeout test uses a slow Painless sub-query (about 1 ms per doc) so the deadline reliably passes partway through
 * the segment. With {@code track_total_hits}, the reported total is the number of docs scored before the search stopped.
 */
public class KNNScoreScriptTimeoutIT extends KNNRestTestCase {

    private static final String INDEX = "knn-score-timeout";
    private static final String FIELD = "vec";
    private static final int DIMENSION = 4;
    private static final int NUM_DOCS = 10_000;
    // Core's first CancellableBulkScorer window. Without the knn_score bulk scorer, a timed-out search scores this many.
    private static final int CORE_FIRST_WINDOW = 4_096;
    private static final float[] QUERY = { 1.0f, 2.0f, 3.0f, 4.0f };

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

    private Map<String, Object> search(String body) throws Exception {
        return toMap(performSearch(INDEX, body));
    }

    private Map<String, Object> toMap(Response response) throws Exception {
        return createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), EntityUtils.toString(response.getEntity())).map();
    }

    @SuppressWarnings("unchecked")
    public void testKnnScore_scoresMatchExpected_withFilterAndBoost() throws Exception {
        createAndLoadIndex();
        final int size = 20;
        final String body = String.format(
            Locale.ROOT,
            "{\"size\":%d,\"query\":{\"script_score\":{\"query\":{\"term\":{\"bucket\":1}},\"script\":%s,\"boost\":2.0}}}",
            size,
            knnScoreScript()
        );

        final Map<String, Object> response = search(body);

        final List<Map<String, Object>> hits = (List<Map<String, Object>>) ((Map<String, Object>) response.get("hits")).get("hits");
        assertEquals(size, hits.size());
        final List<Integer> odd = new ArrayList<>();
        for (int i = 1; i < NUM_DOCS; i += 2) {
            odd.add(i);
        }
        odd.sort(Comparator.comparingDouble((Integer i) -> -expectedScore(i)).thenComparingInt(i -> i));
        final float best = expectedScore(odd.get(0)) * 2.0f;
        assertEquals(best, ((Number) hits.get(0).get("_score")).floatValue(), 1e-5f);
        for (Map<String, Object> hit : hits) {
            final int id = Integer.parseInt((String) hit.get("_id"));
            assertEquals("only bucket 1 docs", 1, id % 2);
            assertEquals("score of doc " + id, expectedScore(id) * 2.0f, ((Number) hit.get("_score")).floatValue(), 1e-5f);
        }
    }

    @SuppressWarnings("unchecked")
    public void testKnnScore_whenTimeoutMidSegment_thenStopsWithinCheckInterval() throws Exception {
        createAndLoadIndex();
        // ~1 ms of Painless work per doc in the sub-query, so the deadline passes within the first few hundred docs.
        final String slowFilter = "{\"script\":{\"script\":{\"source\":\"long s = 0; for (int i = 0; i < params.n; i++) { s += i; } "
            + "return s >= 0;\",\"params\":{\"n\":300000}}}}";
        // size > 0 so the query needs scores; with size 0 it runs as a constant-score query and knn_score never runs.
        final String body = String.format(
            Locale.ROOT,
            "{\"size\":10,\"track_total_hits\":true,\"timeout\":\"10ms\",\"query\":{\"script_score\":{\"query\":{\"bool\":{\"filter\":[%s]}},"
                + "\"script\":%s}}}",
            slowFilter,
            knnScoreScript()
        );

        final Map<String, Object> response = search(body);

        assertEquals(Boolean.TRUE, response.get("timed_out"));
        final int scored = ((Number) ((Map<String, Object>) ((Map<String, Object>) response.get("hits")).get("total")).get("value"))
            .intValue();
        logger.info("knn_score scored [{}] of [{}] docs before the timeout stopped it", scored, NUM_DOCS);
        assertTrue(
            "scored " + scored + " docs; expected fewer than core's first window (" + CORE_FIRST_WINDOW + ")",
            scored < CORE_FIRST_WINDOW
        );
    }
}
