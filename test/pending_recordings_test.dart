import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/pending_recordings.dart';

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('pending_recordings_test');
  });

  tearDown(() {
    if (tempDir.existsSync()) tempDir.deleteSync(recursive: true);
  });

  File writeFakeAudio(String name) {
    final f = File('${tempDir.path}/$name');
    f.writeAsBytesSync([1, 2, 3, 4]);
    return f;
  }

  test('load() on an empty store returns no takes', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();
    expect(store.takes, isEmpty);
  });

  test('add() copies the file and persists it in the manifest', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();

    final source = writeFakeAudio('rec1.wav');
    final take = await store.add('elma', source);

    expect(store.takes, hasLength(1));
    expect(store.takes.first.word, 'elma');
    expect(File(take.filePath).existsSync(), isTrue);
    expect(
        File('${tempDir.path}/pending_recordings/manifest.json').existsSync(),
        isTrue);
  });

  test('a new store instance restores takes saved by a previous one',
      () async {
    final store1 = PendingRecordingsStore(tempDir);
    await store1.load();
    await store1.add('elma', writeFakeAudio('rec1.wav'));
    await store1.add('armut', writeFakeAudio('rec2.wav'));

    final store2 = PendingRecordingsStore(tempDir);
    await store2.load();

    expect(store2.takes, hasLength(2));
    expect(store2.pendingWords, {'elma', 'armut'});
  });

  test('remove() deletes the file and updates the manifest', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();
    final take1 = await store.add('elma', writeFakeAudio('rec1.wav'));
    await store.add('elma', writeFakeAudio('rec2.wav'));

    await store.remove(take1);

    expect(store.takes, hasLength(1));
    expect(File(take1.filePath).existsSync(), isFalse);

    final store2 = PendingRecordingsStore(tempDir);
    await store2.load();
    expect(store2.takes, hasLength(1));
  });

  test('forWord() filters by word', () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();
    await store.add('elma', writeFakeAudio('rec1.wav'));
    await store.add('armut', writeFakeAudio('rec2.wav'));
    await store.add('elma', writeFakeAudio('rec3.wav'));

    expect(store.forWord('elma'), hasLength(2));
    expect(store.forWord('armut'), hasLength(1));
    expect(store.forWord('kayısı'), isEmpty);
  });

  test('add() never overwrites an existing take even under filename collisions',
      () async {
    final store = PendingRecordingsStore(tempDir);
    await store.load();

    final sameInstant = DateTime(2024, 1, 1, 12, 0, 0);
    final take1 =
        await store.add('elma', writeFakeAudio('rec1.wav'), now: sameInstant);
    final take2 =
        await store.add('elma', writeFakeAudio('rec2.wav'), now: sameInstant);

    expect(take1.filePath, isNot(equals(take2.filePath)));
    expect(File(take1.filePath).existsSync(), isTrue);
    expect(File(take2.filePath).existsSync(), isTrue);
    expect(store.takes, hasLength(2));
  });
}
