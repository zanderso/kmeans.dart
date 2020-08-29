// Copyright 2014 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:math';

/// This class describes a clustering.
class Clusters {
  Clusters(this.points, this.clusters, this.means);

  /// The list of all points in the clusters.
  final List<List<double>> points;

  /// For each point `points[i]`, `clusters[i]` gives the index of the cluster
  /// that `points[i]` is a member of. The mean of the cluster is given by
  /// `means[clusters[i]]`.
  final List<int> clusters;

  /// The means of the clusters.
  final List<List<double>> means;

  int get k => means.length;

  /// For each cluster, the list of points belonging to that cluster.
  ///
  /// `clusterPoints[i]` gives the list of points belonging to the cluster with
  /// mean `means[i]`.
  List<List<List<double>>> get clusterPoints {
    if (_clusterPoints != null) {
      return _clusterPoints;
    }
    _clusterPoints = List<List<List<double>>>.generate(
      means.length,
      (int i) => <List<double>>[],
    );
    for (int i = 0; i < points.length; i++) {
      _clusterPoints[clusters[i]].add(points[i]);
    }
    return _clusterPoints;
  }

  List<List<List<double>>> _clusterPoints;

  /// Computes the average 'silhouette' over all points.
  ///
  /// The higher the silhouette the better the clustering.
  ///
  /// This uses an exact computation of the silhouette of each point as defined
  /// in https://en.wikipedia.org/wiki/Silhouette_(clustering).
  double get exactSilhouette {
    if (_exactSilhouette != null) {
      return _exactSilhouette;
    }

    // For each point and each cluster calculate the mean squared distance of
    // the point to each point in the cluster.
    final List<List<double>> meanClusterDist = List<List<double>>.generate(
      means.length,
      (int i) {
        return List<double>.filled(points.length, 0.0);
      },
    );
    for (int k = 0; k < means.length; k++) {
      for (int i = 0; i < points.length; i++) {
        double distSum = 0.0;
        for (int j = 0; j < clusterPoints[k].length; j++) {
          distSum += _distSquared(points[i], clusterPoints[k][j]);
        }
        meanClusterDist[k][i] = clusterPoints[k].length > 1
            ? distSum / (clusterPoints[k].length - 1).toDouble()
            : distSum;
      }
    }

    // For each point find the minimum mean squared distance to a neighbor
    // cluster.
    final List<double> minNeighborDist = List<double>.filled(
      points.length,
      0.0,
    );
    for (int i = 0; i < points.length; i++) {
      double min = double.maxFinite;
      for (int j = 0; j < means.length; j++) {
        if (j != clusters[i] && meanClusterDist[j][i] < min) {
          min = meanClusterDist[j][i];
        }
      }
      minNeighborDist[i] = min;
    }

    // Find the 'silhouette' of each point.
    final List<double> silhouettes = List<double>.filled(points.length, 0.0);
    for (int i = 0; i < points.length; i++) {
      final int c = clusters[i];
      silhouettes[i] = clusterPoints[c].length > 1
          ? (minNeighborDist[i] - meanClusterDist[c][i]) /
              max(minNeighborDist[i], meanClusterDist[c][i])
          : 0.0;
    }

    // Return the average silhouette.
    double silhouetteSum = 0.0;
    for (int i = 0; i < points.length; i++) {
      silhouetteSum += silhouettes[i];
    }
    return _exactSilhouette = silhouetteSum / points.length.toDouble();
  }

  double _exactSilhouette;

  /// Computes an approximation of the average 'silhouette' over all points.
  ///
  /// The higher the silhouette the better the clustering.
  ///
  /// When needed in the silhouette computation, instead of calculating the
  /// average distance of each point to all points in a cluster, this
  /// approximation calculates the average distance to a random subset of at
  /// most 100 points in a cluster.
  ///
  /// See: https://en.wikipedia.org/wiki/Silhouette_(clustering).
  double get silhouette {
    if (_silhouette != null) {
      return _silhouette;
    }
    final Random rng = Random(100);

    // For each point find the mean distance over all points in the nearest
    // neighbor cluster.
    final List<double> minNeighborDist = List<double>.filled(
      points.length,
      0.0,
    );
    for (int i = 0; i < points.length; i++) {
      double minDist = double.maxFinite;
      int minNeighborK = -1;
      for (int k = 0; k < means.length; k++) {
        if (clusters[i] == k) {
          continue;
        }
        final double d = _distSquared(points[i], means[k]);
        if (d < minDist) {
          minDist = d;
          minNeighborK = k;
        }
      }
      // If the size of the cluster is over some threshold, then
      // sample the points isntead of summing all of them.
      final List<List<double>> cluster = clusterPoints[minNeighborK];
      final int neighborSize = cluster.length;
      if (neighborSize <= 100) {
        double sum = 0.0;
        for (int j = 0; j < cluster.length; j++) {
          sum += _distSquared(points[i], cluster[j]);
        }
        minNeighborDist[i] = sum / cluster.length.toDouble();
      } else {
        // 100 samples.
        double sum = 0.0;
        for (int j = 0; j < 100; j++) {
          final int sample = rng.nextInt(neighborSize);
          sum += _distSquared(points[i], cluster[sample]);
        }
        minNeighborDist[i] = sum / 100.0;
      }
    }

    // For each point find the mean distance over all points in the same
    // cluster.
    final List<double> meanClusterDist = List<double>.filled(
      points.length,
      0.0,
    );
    for (int i = 0; i < points.length; i++) {
      // If the size of the cluster is over some threshold, then
      // sample the points isntead of summing all of them.
      final int k = clusters[i];
      final int clusterSize = clusterPoints[k].length;
      if (clusterSize <= 100) {
        double sum = 0.0;
        for (int j = 0; j < clusterPoints[k].length; j++) {
          sum += _distSquared(points[i], clusterPoints[k][j]);
        }
        meanClusterDist[i] = clusterPoints[k].length > 1
            ? sum / (clusterPoints[k].length - 1).toDouble()
            : sum;
      } else {
        double sum = 0.0;
        for (int j = 0; j < 100; j++) {
          final int sample = rng.nextInt(clusterSize);
          sum += _distSquared(points[i], clusterPoints[k][sample]);
        }
        meanClusterDist[i] = sum / 100;
      }
    }

    // Find the 'silhouette' of each point.
    final List<double> silhouettes = List<double>.filled(points.length, 0.0);
    for (int i = 0; i < points.length; i++) {
      final int c = clusters[i];
      silhouettes[i] = clusterPoints[c].length > 1
          ? (minNeighborDist[i] - meanClusterDist[i]) /
              max(minNeighborDist[i], meanClusterDist[i])
          : 0.0;
    }

    // Return the average silhouette.
    double silhouetteSum = 0.0;
    for (int i = 0; i < points.length; i++) {
      silhouetteSum += silhouettes[i];
    }
    return _silhouette = silhouetteSum / points.length.toDouble();
  }

  double _silhouette;

  @override
  String toString() {
    final StringBuffer buf = StringBuffer();
    buf.writeln('Means:');
    for (int i = 0; i < means.length; i++) {
      buf.writeln(means[i]);
    }
    buf.writeln('');
    buf.writeln('Points');
    for (int i = 0; i < points.length; i++) {
      buf.writeln('${points[i]} -> ${clusters[i]}');
    }
    return buf.toString();
  }
}

/// A class for organizing k-means computations with utility methods for
/// finding a good solution.
///
/// https://en.wikipedia.org/wiki/K-means_clustering
///
/// Uses k-means++ (https://en.wikipedia.org/wiki/K-means%2B%2B) for the
/// initial means, and then iterates.
///
/// The points accepted by [KMeans] can have any number of dimensions and are
/// represented by `List<double>`. The distance function is hard-coded to be
/// the euclidan distance.
class KMeans {
  KMeans(this.points);

  static const double _kPrecision = 1e-6;

  /// The points to cluster.
  final List<List<double>> points;

  /// Returns a [Cluster] of [points] into [k] clusters.
  Clusters compute(
    int k, {
    int maxIterations = 10,
    int seed = 42,
  }) {
    final List<int> clusters = List<int>.filled(points.length, 0);
    final List<List<double>> means = _kMeansPlusPlusInit(points, k, seed);

    for (int iters = 0; iters < maxIterations; iters++) {
      // Put points into the closest cluster.
      _populateClusters(means, clusters);

      // Shift means.
      if (!_shiftClusters(clusters, means)) {
        break;
      }
    }

    return Clusters(points, clusters, means);
  }

  /// Finds the 'best' k-means [Cluster] over a range of possible values of
  /// 'k'.
  ///
  /// Returns the clustering whose 'k' maximmizes the average 'silhouette'
  /// (https://en.wikipedia.org/wiki/Silhouette_(clustering)).
  Clusters computeBest({
    int maxIterations = 10,
    int seed = 42,
    int minK = 2,
    int maxK = 20,
    int trialsPerK = 1,
  }) {
    Clusters best;
    int k = minK;
    int trial = 0;

    while (k <= maxK && k <= points.length) {
      final Clusters km = compute(
        k,
        maxIterations: maxIterations,
        seed: seed + trial,
      );
      if (best == null || km.silhouette > best.silhouette) {
        best = km;
      }
      trial++;
      if (trial == trialsPerK) {
        k++;
        trial = 0;
      }
    }

    return best;
  }

  // Returns an initial set of k means initialized according to the k-means++
  // algorithm:
  // https://en.wikipedia.org/wiki/K-means%2B%2B
  List<List<double>> _kMeansPlusPlusInit(
    List<List<double>> points,
    int k,
    int seed,
  ) {
    final Random rng = Random(seed);
    final int dim = points[0].length;
    final List<List<double>> means = List<List<double>>.generate(k, (int i) {
      return List<double>.filled(dim, 0.0);
    });

    means[0] = points[rng.nextInt(points.length)];
    for (int i = 1; i < k; i++) {
      final List<double> ds = points.map((List<double> p) {
        double minDist = double.maxFinite;
        for (int j = 0; j < i; j++) {
          final double d = _distSquared(means[j], p);
          if (d < minDist) {
            minDist = d;
          }
        }
        return minDist;
      }).toList();
      final double sum = ds.fold(0.0, (double a, double b) => a + b);
      final List<double> ps = ds.map((double x) => x / sum).toList();
      int pointIndex = 0;
      final double r = rng.nextDouble();
      double cum = 0.0;
      while (true) {
        cum += ps[pointIndex];
        if (cum > r) {
          break;
        }
        pointIndex++;
      }
      means[i] = points[pointIndex];
    }

    return means;
  }

  // Put points into the closest cluster.
  void _populateClusters(List<List<double>> means, List<int> outClusters) {
    for (int i = 0; i < points.length; i++) {
      int kidx = 0;
      double kdist = _distSquared(points[i], means[0]);
      for (int j = 1; j < means.length; j++) {
        final double dist = _distSquared(points[i], means[j]);
        if (dist < kdist) {
          kidx = j;
          kdist = dist;
        }
      }
      outClusters[i] = kidx;
    }
  }

  // Shift cluster means.
  bool _shiftClusters(List<int> clusters, List<List<double>> means) {
    bool shifted = false;
    final int dim = means[0].length;
    for (int i = 0; i < means.length; i++) {
      final List<double> newMean = List<double>.filled(dim, 0.0);
      double n = 0.0;
      for (int j = 0; j < points.length; j++) {
        // TODO(zra): Only iterate over the points in the right cluster.
        if (clusters[j] != i) {
          continue;
        }
        for (int m = 0; m < dim; m++) {
          newMean[m] = (newMean[m] * n + points[j][m]) / (n + 1.0);
        }
        n += 1.0;
      }
      for (int m = 0; m < dim; m++) {
        if ((means[i][m] - newMean[m]).abs() > _kPrecision) {
          shifted = true;
          break;
        }
      }
      means[i] = newMean;
    }
    return shifted;
  }
}

double _distSquared(List<double> a, List<double> b) {
  assert(a.length == b.length);

  final int length = a.length;
  double sum = 0.0;
  for (int i = 0; i < length; i++) {
    final double diff = a[i] - b[i];
    final double diffSquared = diff * diff;
    sum += diffSquared;
  }

  return sum;
}
