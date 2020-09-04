# package:kmeans

An implementation of [k-means clustering][kmeans] for Dart.

k-means tends to be inflexible, preferring to produce equal-sized clusters. See
the wikipedia article for full details.

In this implementation, data points are translated and scaled in each dimension
to try to give each parameter equal weight.

By default, clusters are initialized using the [k-means++ method][kmeans++].
However, repeated trials are still recommended to avoid local minima. The user
can also supply an initial set of means.

To evaluate the goodness of a clustering, and to [find good values for k][determining]
when one is not known in advance, this library includes a calculation of the
[silhouette][silhouette] of a clustering, and uses it in the `bestFit()` method.
The

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

[determining]: https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
[kmeans-pub]: https://pub.dev/packages/kmeans
[kmeans]: https://en.wikipedia.org/wiki/K-means_clustering
[kmeans++]: https://en.wikipedia.org/wiki/K-means%2B%2B
[silhouette]: https://en.wikipedia.org/wiki/Silhouette_(clustering)
[tracker]: https://github.com/zanderso/kmeans.dart/issues
