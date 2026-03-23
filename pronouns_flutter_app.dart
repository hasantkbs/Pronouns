import 'package:flutter/material.dart';

void main() {
  runApp(PronounsApp());
}

class PronounsApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pronouns AI',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Pronouns AI')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.count(
          crossAxisCount: 2,
          crossAxisSpacing: 16,
          mainAxisSpacing: 16,
          children: [
            _FeatureCard(
              title: 'Ses Kaydı',
              icon: Icons.mic,
              color: Colors.blue,
              onTap: () => _navigateTo(context, 'Recording'),
            ),
            _FeatureCard(
              title: 'Model Eğitimi',
              icon: Icons.model_training,
              color: Colors.orange,
              onTap: () => _navigateTo(context, 'Training'),
            ),
            _FeatureCard(
              title: 'Konuşma Çevirme',
              icon: Icons.translate,
              color: Colors.green,
              onTap: () => _navigateTo(context, 'Translation'),
            ),
            _FeatureCard(
              title: 'Ayarlar',
              icon: Icons.settings,
              color: Colors.grey,
              onTap: () => _navigateTo(context, 'Settings'),
            ),
          ],
        ),
      ),
    );
  }

  void _navigateTo(BuildContext context, String feature) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('$feature özelliği yakında eklenecek!')),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _FeatureCard({
    required this.title,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        decoration: BoxDecoration(
          color: color.withOpacity(0.2),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color, width: 2),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 48, color: color),
            SizedBox(height: 12),
            Text(
              title,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}
