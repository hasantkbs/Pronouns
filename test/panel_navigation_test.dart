import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/main.dart';

void main() {
  testWidgets('Furkanca button opens a full page, not a bottom sheet',
      (WidgetTester tester) async {
    // Varsayılan test yüzeyi (800x600) gerçek telefon ekranlarından kısa
    // olduğu için ana sayfa taşabilir; telefon boyutuna benzer bir yüzey
    // ayarlanıyor.
    await tester.binding.setSurfaceSize(const Size(400, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    // Ana sayfadaki logo sonsuz döngüde "pulse" animasyonu oynattığı için
    // pumpAndSettle() hiç durmaz; bunun yerine sabit pump() adımları kullanılır.
    await tester.pumpWidget(const PronounsApp());
    await tester.pump();

    // 'Furkanca' metni hem başlıkta hem buton etiketinde geçtiği için
    // butona özgü alt metinden hedefleniyor.
    await tester.tap(find.text('Konuş → Düzeltilmiş metin & ses'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    // Tam sayfa push edildiğinde geri oklu bir AppBar görünür.
    expect(find.byType(AppBar), findsOneWidget);
    expect(find.byTooltip('Back'), findsOneWidget);
    // Modal bottom sheet olsaydı BottomSheet widget'ı bulunurdu.
    expect(find.byType(BottomSheet), findsNothing);
  });
}
