A simple implementation of k-means clustering.

Clusters are initialized using the [k-means++ method][kmeans++].

To evaluate the goodness of a clustering, and to find good values for k when
one is not known a priori, this library includes a calculation of the
[silhouette][silhouette] of a clustering.

## Usage

A simple usage example:

```dart
import 'package:kmeans/kmeans.dart';

main() {
  var kmeans = KMeans([
    [0.0, 0.0], [1.1, 1.1], [-5.0, -0.50], ...
  ]);
  var k = 3;
  var clusters = kmeans.compute(k);
  var silhouette = clusters.silhouette;
  print('The clusters have silhouette $silhouette');
  for (int i = 0; i < kmeans.points.length; i++) {
    var point = kmeans.points[i];
    var cluster = kmeans.clusters[i];
    var mean = kmeans.means[cluster];
    print('$point is in cluster $cluster with mean $mean.');
  }
  ...
  var bestCluster = kmeans.computeBest(
    minK: 3,
    maxK: 10,
  );
}
```

## Features and bugs

Please file feature requests and bugs at the [issue tracker][tracker].

[kmeans++]: https://en.wikipedia.org/wiki/K-means%2B%2B
[silhouette]: https://en.wikipedia.org/wiki/Silhouette_(clustering)
[tracker]: https://github.com/zanderso/kmeans.dart/issues
