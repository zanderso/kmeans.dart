// Copyright 2014 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:math';

import 'package:kmeans/kmeans.dart';
import 'package:test/test.dart';

void main() {
  test('KMeans sanity check', () {
    final List<List<double>> points = <List<double>>[
      <double>[0.9],
      <double>[1.0],
      <double>[1.1],
      <double>[1.9],
      <double>[2.0],
      <double>[2.1],
    ];
    final KMeans kmeans = KMeans(points);
    final Clusters k = kmeans.compute(2);

    expect(k.means.length, equals(2));
    expect(k.means[0] != k.means[1], isTrue);
    expect(k.clusters[0] == k.clusters[1], isTrue);
    expect(k.clusters[1] == k.clusters[2], isTrue);
    expect(k.clusters[2] != k.clusters[3], isTrue);
    expect(k.clusters[3] == k.clusters[4], isTrue);
    expect(k.clusters[4] == k.clusters[5], isTrue);

    expect(k.clusterPoints[0].length, equals(3));
    expect(k.clusterPoints[1].length, equals(3));
  });

  test('Best KMeans sanity check', () {
    final List<List<double>> points = <List<double>>[
      <double>[0.9],
      <double>[1.0],
      <double>[1.1],
      <double>[1.9],
      <double>[2.0],
      <double>[2.1],
      <double>[2.9],
      <double>[3.0],
      <double>[3.1],
    ];
    final KMeans kmeans = KMeans(points);
    final Clusters k = kmeans.computeBest();

    expect(k.means.length, equals(3));
    expect(k.means[0] != k.means[1], isTrue);
    expect(k.means[1] != k.means[2], isTrue);
    expect(k.clusters[0] == k.clusters[1], isTrue);
    expect(k.clusters[1] == k.clusters[2], isTrue);
    expect(k.clusters[2] != k.clusters[3], isTrue);
    expect(k.clusters[3] == k.clusters[4], isTrue);
    expect(k.clusters[4] == k.clusters[5], isTrue);
    expect(k.clusters[5] != k.clusters[6], isTrue);
    expect(k.clusters[6] == k.clusters[7], isTrue);
    expect(k.clusters[7] == k.clusters[8], isTrue);
    expect(k.clusterPoints[0].length, equals(3));
    expect(k.clusterPoints[1].length, equals(3));
    expect(k.clusterPoints[2].length, equals(3));
  });

  List<List<double>> initializePoints(
    int clusters,
    int pointsPerCluster,
    double maxX,
    double maxY,
    Random rng,
  ) {
    final List<Point<double>> means = List<Point<double>>.generate(
      clusters,
      (int i) {
        return Point<double>(
          rng.nextDouble() * 1000.0,
          rng.nextDouble() * 1000.0,
        );
      },
    );

    final List<double> sds = List<double>.generate(clusters, (int i) {
      return rng.nextDouble();
    });

    double nextGaussian() {
      double v1, v2, s;
      do {
        v1 = 2.0 * rng.nextDouble() - 1.0;
        v2 = 2.0 * rng.nextDouble() - 1.0;
        s = v1 * v1 + v2 * v2;
      } while (s >= 1.0 || s == 0.0);
      s = sqrt(-2.0 * log(s) / s);
      return v1 * s;
    }

    final List<List<double>> points = <List<double>>[];
    for (int i = 0; i < clusters * pointsPerCluster; i++) {
      final int cluster = i ~/ pointsPerCluster;
      final Point<double> mean = means[cluster];
      final double sd = sds[cluster];
      final double x = mean.x + nextGaussian() * sd;
      final double y = mean.y + nextGaussian() * sd;
      points.add(<double>[x, y]);
    }

    return points;
  }

  test('Generated clusters best k-means finds the right k', () {
    const int minK = 2;
    const int maxK = 15;
    const int pointsPerCluster = 100;
    final Random rng = Random(42);

    for (int clusters = minK; clusters < maxK; clusters++) {
      final List<List<double>> points = initializePoints(
        clusters,
        pointsPerCluster,
        1000.0,
        1000.0,
        rng,
      );

      final KMeans kmeans = KMeans(points);
      final Clusters k = kmeans.computeBest(
        minK: minK,
        maxK: maxK,
      );

      // We found the right k.
      expect(k.k, equals(clusters));
    }
  });

  test('The approximate silhouette is close to the exact silhouette', () {
    const int minK = 2;
    const int maxK = 10;
    const int pointsPerCluster = 200;
    final Random rng = Random(42);

    for (int clusters = minK; clusters <= maxK; clusters++) {
      final List<List<double>> points = initializePoints(
        clusters,
        pointsPerCluster,
        1000.0,
        1000.0,
        rng,
      );

      final KMeans kmeans = KMeans(points);
      for (int i = minK; i <= maxK; i++) {
        final Clusters c = kmeans.compute(i);
        expect((c.silhouette - c.exactSilhouette).abs(), lessThan(0.02));
      }
    }
  });
}
