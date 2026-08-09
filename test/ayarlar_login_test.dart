import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pronouns_ai/main.dart';

void main() {
  testWidgets('Ayarlar butonu şifre olmadan doğrudan açılmaz, yanlış girişte hata gösterir',
      (WidgetTester tester) async {
    // MainPage'deki sonsuz "pulse" animasyonu pumpAndSettle() ile takılır;
    // sabit pump() adımları kullanılıyor (bkz. panel_navigation_test.dart).
    await tester.binding.setSurfaceSize(const Size(400, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await tester.pumpWidget(const PronounsApp());
    await tester.pump();

    await tester.tap(find.text('Ayarlar'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    // Doğrudan Ayarlar sayfasına değil, giriş ekranına gitmeli.
    expect(find.text('Ayarlar Girişi'), findsOneWidget);
    expect(find.text('Model Ayarları'), findsNothing);

    // Yanlış kullanıcı adı/şifre.
    await tester.enterText(find.byType(TextField).first, 'admin');
    await tester.enterText(find.byType(TextField).last, 'yanlis-sifre');
    await tester.tap(find.text('Giriş Yap'));
    await tester.pump();

    expect(find.text('Kullanıcı adı veya şifre hatalı'), findsOneWidget);
    // Hâlâ giriş ekranındayız — Ayarlar sayfasına geçmedi.
    expect(find.text('Ayarlar Girişi'), findsOneWidget);
  });
}
