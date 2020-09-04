// Copyright 2014 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:io';
import 'dart:math';

import 'package:kmeans/kmeans.dart';
import 'package:test/test.dart';

import 'utils/arff.dart';

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
    final Clusters k = kmeans.fit(2);

    expect(k.k, equals(2));
    expect(k.means.length, equals(2));
    expect(k.means[0] != k.means[1], isTrue);
    expect(k.clusters[0] == k.clusters[1], isTrue);
    expect(k.clusters[1] == k.clusters[2], isTrue);
    expect(k.clusters[2] != k.clusters[3], isTrue);
    expect(k.clusters[3] == k.clusters[4], isTrue);
    expect(k.clusters[4] == k.clusters[5], isTrue);

    expect(k.clusterPoints[0].length, equals(3));
    expect(k.clusterPoints[1].length, equals(3));

    expect(
        (k.inertia - 0.04).abs(), lessThanOrEqualTo(KMeans.defaultPrecision));

    expect(
        k.predict(<List<double>>[
          <double>[k.means[0][0] + 0.05]
        ]),
        equals(<int>[0]));
    expect(
        k.predict(<List<double>>[
          <double>[k.means[1][0] + 0.05]
        ]),
        equals(<int>[1]));
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
    final Clusters k = kmeans.bestFit();

    expect(k.k, equals(3));
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

    expect(
        (k.inertia - 0.06).abs(), lessThanOrEqualTo(KMeans.defaultPrecision));

    expect(
      k.predict(<List<double>>[
        <double>[k.means[0][0] + 0.05],
        <double>[k.means[1][0] + 0.05],
        <double>[k.means[2][0] + 0.05],
      ]),
      equals(<int>[0, 1, 2]),
    );
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
      final Clusters k = kmeans.bestFit(
        minK: minK,
        maxK: maxK,
      );

      // We found the right k.
      expect(k.k, equals(clusters));
      expect(k.means.length, equals(clusters));
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
        final Clusters c = kmeans.fit(i);
        expect((c.silhouette - c.exactSilhouette).abs(), lessThan(0.02));
      }
    }
  });

  test2DArff(
    fileName: '2d-10c',
    k: 9,
    minK: 2,
    maxK: 15,
    trials: 3,
    errors: 5,
  );

  test2DArff(
    fileName: '2d-20c-no0',
    k: 20,
    minK: 18,
    maxK: 21,
    trials: 100,
    errors: 50,
  );

  test2DArff(
    fileName: '2d-4c-no9',
    k: 4,
    minK: 2,
    maxK: 6,
    errors: 60,
    trials: 1,
  );
}

void test2DArff(
    {String fileName, int k, int errors, int minK, int maxK, int trials}) {
  test('$fileName with labels', () async {
    final ArffReader reader =
        ArffReader.fromFile(File('test/data/$fileName.arff'));
    await reader.parse();

    // The third dimension is the label. Exagerate it to test that bestFit is
    // finding the right number of clusters and getting all the right points
    // into those clusters.
    for (int i = 0; i < reader.data.length; i++) {
      reader.data[i][2] *= 1e6;
    }

    final KMeans kmeans = KMeans(reader.data, ignoredDims: <int>[]);
    final Clusters clusters = kmeans.bestFit(
      minK: minK,
      maxK: maxK,
      trialsPerK: trials,
    );

    expect(clusters.k, equals(k));

    // The third dimension of each point gives the actual cluster. Check that
    // all points we clustered together should really be together.
    for (int i = 0; i < clusters.clusterPoints.length; i++) {
      final double c = clusters.clusterPoints[i][0][2];
      for (int j = 0; j < clusters.clusterPoints[i].length; j++) {
        expect(clusters.clusterPoints[i][j][2], equals(c));
      }
    }
  });

  test('$fileName without labels', () async {
    final ArffReader reader =
        ArffReader.fromFile(File('test/data/$fileName.arff'));
    await reader.parse();

    // Ignore the third dimension of each point. It is the label, which we
    // leave off now, and then check later.
    final KMeans kmeans = KMeans(reader.data, ignoredDims: <int>[2]);
    final Clusters clusters = kmeans.bestFit(
      minK: minK,
      maxK: maxK,
      trialsPerK: trials,
    );

    expect(clusters.k, equals(k));

    // May get a few wrong.
    int errors = 0;
    for (int i = 0; i < clusters.clusterPoints.length; i++) {
      final List<int> counts = List<int>.filled(clusters.k + 1, 0);
      for (int j = 0; j < clusters.clusterPoints[i].length; j++) {
        counts[clusters.clusterPoints[i][j][2].toInt()]++;
      }
      int maxIdx = -1;
      int maxCount = 0;
      for (int j = 0; j < clusters.k + 1; j++) {
        if (counts[j] > maxCount) {
          maxCount = counts[j];
          maxIdx = j;
        }
      }
      // If a point isn't in the maxIdx cluster, consider it an error.
      for (int j = 0; j < clusters.clusterPoints[i].length; j++) {
        if (clusters.clusterPoints[i][j][2].toInt() != maxIdx) {
          errors++;
        }
      }
    }
    expect(errors, lessThanOrEqualTo(errors));
  });
}
