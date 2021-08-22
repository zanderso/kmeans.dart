// Copyright 2014 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:io';

import 'package:kmeans/kmeans.dart';

import '../test/utils/arff.dart';

Future<void> main() async {
  // See example/letter.arff for an explanation of the data.
  final ArffReader reader = ArffReader.fromFile(File('example/letter.arff'));
  await reader.parse();

  const int labelIndex = 16;
  final int trainingSetSize = reader.data!.length ~/ 2;

  // Train on the first half of the data.
  final List<List<double>> trainingData =
      reader.data!.sublist(0, trainingSetSize);
  final KMeans trainingKMeans = KMeans(trainingData, labelDim: labelIndex);
  final Clusters trainingClusters = trainingKMeans.bestFit(
    minK: 26,
    maxK: 26,
  )!;

  int trainingErrors = 0;
  for (int i = 0; i < trainingClusters.clusterPoints!.length; i++) {
    // Count the occurrences of each label.
    final List<int> counts = List<int>.filled(trainingClusters.k, 0);
    for (int j = 0; j < trainingClusters.clusterPoints![i].length; j++) {
      counts[trainingClusters.clusterPoints![i][j][labelIndex].toInt()]++;
    }

    // Find the most frequent label.
    int maxIdx = -1;
    int maxCount = 0;
    for (int j = 0; j < trainingClusters.k; j++) {
      if (counts[j] > maxCount) {
        maxCount = counts[j];
        maxIdx = j;
      }
    }

    // If a point isn't in the most frequent cluster, consider it an error.
    for (int j = 0; j < trainingClusters.clusterPoints![i].length; j++) {
      if (trainingClusters.clusterPoints![i][j][labelIndex].toInt() != maxIdx) {
        trainingErrors++;
      }
    }
  }
  // This should be zero.
  print('trainingErrors: $trainingErrors');

  // A copy of the clustering that ignores the labels.
  final Clusters ignoreLabel = Clusters(
    trainingClusters.points,
    <int>[labelIndex],
    trainingClusters.clusters,
    trainingClusters.means,
  );

  // Find the predicted cluster for the other half of the data.
  final List<List<double>> data = reader.data!.sublist(trainingSetSize);
  final List<int> predictions = data.map((List<double> point) {
    return ignoreLabel.kNearestNeighbors(point, 5);
  }).toList();

  // Count the number points where the predicted cluster doesn't match the
  // label.
  int errors = 0;
  for (int i = 0; i < predictions.length; i++) {
    final List<double> rep = ignoreLabel.clusterPoints![predictions[i]][0];
    final int repClass = rep[labelIndex].toInt();
    final int dataClass = data[i][labelIndex].toInt();
    if (dataClass != repClass) {
      errors++;
    }
  }

  // Hopefully these are small.
  final int testSetSize = reader.data!.length - trainingSetSize;
  final double errorRate = errors.toDouble() / testSetSize.toDouble();
  print('classification errors: $errors / $testSetSize');
  print('error rate: $errorRate');
}
