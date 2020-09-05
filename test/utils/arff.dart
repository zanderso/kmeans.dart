// Copyright 2014 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:convert';
import 'dart:io';

class ArffReader {
  ArffReader.fromFile(File f) : _dataStream = f.openRead();

  ArffReader.fromStream(this._dataStream);

  final Stream<List<int>> _dataStream;

  List<List<double>> get data => _data;
  List<List<double>> _data;

  Future<void> parse() async {
    final List<List<double>> dataList = <List<double>>[];
    bool parsingData = false;

    final Stream<String> lineStream =
        _dataStream.transform(utf8.decoder).transform(const LineSplitter());
    await for (final String l in lineStream) {
      final String line = l.trim();
      if (line.startsWith('%')) {
        continue;
      }
      if (line.contains('@data') || line.contains('@DATA')) {
        parsingData = true;
        continue;
      }
      if (!parsingData) {
        continue;
      }
      final List<String> sampleStrings = line.split(',');
      final List<double> sampleNumbers = sampleStrings.map((String d) {
        return double.tryParse(d) ?? _stringTableIndex(d).toDouble();
      }).toList();
      dataList.add(sampleNumbers);
    }
    _data = dataList;
  }

  final Map<String, int> _stringTable = <String, int>{};
  int _stringTableIndex(String s) {
    final int length = _stringTable.length;
    return _stringTable.putIfAbsent(s, () => length);
  }
}
