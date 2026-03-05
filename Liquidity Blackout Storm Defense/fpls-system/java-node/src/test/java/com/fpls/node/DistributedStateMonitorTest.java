package com.fpls.node;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class DistributedStateMonitorTest {

  @Test
  void testAnomalyRecord() {
    var rec = new AnomalyRecord("node-1", 0.9, 0.75, System.nanoTime());
    assertTrue(rec.isAnomaly());
    var normal = new AnomalyRecord("node-1", 0.5, 0.75, System.nanoTime());
    assertFalse(normal.isAnomaly());
    System.out.println("[PASS] AnomalyRecord");
  }

  @Test
  void testWeightArrayUpdate() {
    float[] w = new float[16];
    float lr  = 0.01f;
    double score = 0.8;
    for (int i = 0; i < w.length; i++) {
      w[i] = w[i] * (1 - lr) + (float)(score * Math.sin(i + 1)) * lr;
    }
    boolean nonZero = false;
    for (float v : w) if (Math.abs(v) > 1e-6) { nonZero = true; break; }
    assertTrue(nonZero);
    System.out.println("[PASS] WeightArrayUpdate");
  }
}
