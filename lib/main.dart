import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(PronounsApp());
}

class PronounsApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pronouns AI',
      debugShowCheckedModeBanner: false,
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
  final String userId = 'FurkanV1';
  final String baseUrl = 'http://91.241.50.187:8001';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Pronouns AI - $userId'),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _FeatureCard(
                title: 'Ses Kaydı',
                subtitle: 'Model eğitimi için veri topla',
                icon: Icons.mic,
                color: Colors.blueAccent,
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => RecordingScreen(userId: userId, baseUrl: baseUrl)),
                ),
              ),
              SizedBox(height: 24),
              _FeatureCard(
                title: 'Çeviri Yap',
                subtitle: 'Konuşmayı metne/sese çevir',
                icon: Icons.translate,
                color: Colors.greenAccent,
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => TranslationScreen(userId: userId, baseUrl: baseUrl)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _FeatureCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        width: double.infinity,
        padding: EdgeInsets.all(32),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [color.withOpacity(0.7), color.withOpacity(0.4)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color, width: 2),
          boxShadow: [
            BoxShadow(color: color.withOpacity(0.3), blurRadius: 10, offset: Offset(0, 4)),
          ],
        ),
        child: Column(
          children: [
            Icon(icon, size: 64, color: Colors.white),
            SizedBox(height: 16),
            Text(
              title,
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
            ),
            SizedBox(height: 8),
            Text(
              subtitle,
              style: TextStyle(fontSize: 16, color: Colors.white70),
            ),
          ],
        ),
      ),
    );
  }
}

// --- Recording Screen ---
class RecordingScreen extends StatefulWidget {
  final String userId;
  final String baseUrl;

  RecordingScreen({required this.userId, required this.baseUrl});

  @override
  _RecordingScreenState createState() => _RecordingScreenState();
}

class _RecordingScreenState extends State<RecordingScreen> {
  List<dynamic> words = [];
  bool isLoading = true;
  final record = AudioRecorder();
  final audioPlayer = AudioPlayer();
  
  bool isRecording = false;
  bool hasRecorded = false;
  String? currentWord;
  String? recordedFilePath;
  bool isUploading = false;

  @override
  void initState() {
    super.initState();
    _fetchWords();
  }

  Future<void> _fetchWords() async {
    try {
      final response = await http.get(Uri.parse('${widget.baseUrl}/words?user_id=${widget.userId}'));
      if (response.statusCode == 200) {
        setState(() {
          words = json.decode(response.body)['words'];
          isLoading = false;
        });
      }
    } catch (e) {
      print('Error fetching words: $e');
    }
  }

  Future<void> _startRecording(String word) async {
    if (await Permission.microphone.request().isGranted) {
      final directory = await getTemporaryDirectory();
      final path = '${directory.path}/temp_word_${DateTime.now().millisecondsSinceEpoch}.wav';
      
      await record.start(const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1), path: path);
      setState(() {
        isRecording = true;
        hasRecorded = false;
        currentWord = word;
        recordedFilePath = null;
      });
    }
  }

  Future<void> _stopRecording() async {
    final path = await record.stop();
    setState(() {
      isRecording = false;
      hasRecorded = true;
      recordedFilePath = path;
    });
  }

  Future<void> _playRecorded() async {
    if (recordedFilePath != null) {
      await audioPlayer.play(DeviceFileSource(recordedFilePath!));
    }
  }

  Future<void> _uploadFile() async {
    if (recordedFilePath == null || currentWord == null) return;

    setState(() => isUploading = true);
    
    var request = http.MultipartRequest('POST', Uri.parse('${widget.baseUrl}/record'));
    request.fields['user_id'] = widget.userId;
    request.fields['word'] = currentWord!;
    request.files.add(await http.MultipartFile.fromPath('audio', recordedFilePath!));

    try {
      final response = await request.send();
      if (response.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('"$currentWord" başarıyla yüklendi.')));
        setState(() {
          hasRecorded = false;
          currentWord = null;
          recordedFilePath = null;
          isUploading = false;
        });
        _fetchWords(); // Refresh list (completed words will disappear)
      }
    } catch (e) {
      setState(() => isUploading = false);
      print('Upload error: $e');
    }
  }
@override
Widget build(BuildContext context) {
  int completedWords = words.where((w) => w['count'] >= w['required']).length; // This logic is simplified as backend filters them, but good for local UI state if any

  return Scaffold(
    appBar: AppBar(
      title: Text('Ses Kaydı'),
      actions: [
        if (!isLoading && words.isNotEmpty)
          TextButton.icon(
            onPressed: () => Navigator.pop(context),
            icon: Icon(Icons.check, color: Colors.greenAccent),
            label: Text('KAYDI BİTİR', style: TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold)),
          ),
      ],
    ),
    body: isLoading
        ? Center(child: CircularProgressIndicator())
        : Column(
            children: [
              Container(
                width: double.infinity,
                padding: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                color: Colors.white10,
                child: Text(
                  'Kalan Kelime Sayısı: ${words.length}',
                  style: TextStyle(color: Colors.orangeAccent, fontWeight: FontWeight.w500),
                  textAlign: TextAlign.center,
                ),
              ),
              if (isRecording || hasRecorded)
                Container(
...
                    width: double.infinity,
                    padding: EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.blueGrey.withOpacity(0.2),
                      border: Border(bottom: BorderSide(color: Colors.blueAccent)),
                    ),
                    child: Column(
                      children: [
                        Text(isRecording ? 'KAYDEDİLİYOR' : 'KAYIT TAMAMLANDI', 
                             style: TextStyle(color: isRecording ? Colors.redAccent : Colors.greenAccent, fontWeight: FontWeight.bold)),
                        SizedBox(height: 8),
                        Text('"$currentWord"', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
                        SizedBox(height: 16),
                        if (isRecording)
                          ElevatedButton.icon(
                            onPressed: _stopRecording,
                            icon: Icon(Icons.stop),
                            label: Text('DURDUR'),
                            style: ElevatedButton.styleFrom(backgroundColor: Colors.red, foregroundColor: Colors.white),
                          ),
                        if (hasRecorded)
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              IconButton(
                                icon: Icon(Icons.play_arrow, size: 40, color: Colors.blueAccent),
                                onPressed: _playRecorded,
                                tooltip: 'Dinle',
                              ),
                              SizedBox(width: 16),
                              IconButton(
                                icon: Icon(Icons.refresh, size: 40, color: Colors.orangeAccent),
                                onPressed: () => _startRecording(currentWord!),
                                tooltip: 'Tekrar Kaydet',
                              ),
                              SizedBox(width: 16),
                              if (isUploading) 
                                CircularProgressIndicator()
                              else
                                IconButton(
                                  icon: Icon(Icons.cloud_upload, size: 40, color: Colors.greenAccent),
                                  onPressed: _uploadFile,
                                  tooltip: 'Kaydı Gönder',
                                ),
                            ],
                          ),
                      ],
                    ),
                  ),
                Expanded(
                  child: words.isEmpty 
                    ? Center(child: Text('Tüm kayıtlar tamamlandı!'))
                    : ListView.builder(
                        itemCount: words.length,
                        itemBuilder: (context, index) {
                          final item = words[index];
                          return ListTile(
                            title: Text(item['word'], style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                            subtitle: Text('Eksik Kayıt: ${item['required'] - item['count']} / ${item['required']}'),
                            trailing: IconButton(
                              icon: Icon(Icons.mic, color: Colors.blueAccent),
                              onPressed: (isRecording || isUploading) ? null : () => _startRecording(item['word']),
                            ),
                          );
                        },
                      ),
                ),
              ],
            ),
    );
  }
}

// --- Translation Screen ---
class TranslationScreen extends StatefulWidget {
  final String userId;
  final String baseUrl;

  TranslationScreen({required this.userId, required this.baseUrl});

  @override
  _TranslationScreenState createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  final record = AudioRecorder();
  final audioPlayer = AudioPlayer();
  bool isRecording = false;
  String status = 'Hazır';
  String resultText = '';
  String responseText = '';
  bool isProcessing = false;

  Future<void> _startTranslation() async {
    if (await Permission.microphone.request().isGranted) {
      final directory = await getTemporaryDirectory();
      final path = '${directory.path}/temp_translate_${DateTime.now().millisecondsSinceEpoch}.wav';
      
      await record.start(const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1), path: path);
      setState(() {
        isRecording = true;
        status = 'Dinliyor...';
        resultText = '';
        responseText = '';
      });
      
      Timer(Duration(seconds: 4), () {
        if (isRecording) _stopAndTranslate();
      });
    }
  }

  Future<void> _stopAndTranslate() async {
    final path = await record.stop();
    setState(() {
      isRecording = false;
      isProcessing = true;
      status = 'İşleniyor...';
    });

    if (path != null) {
      var request = http.MultipartRequest('POST', Uri.parse('${widget.baseUrl}/translate'));
      request.fields['user_id'] = widget.userId;
      request.files.add(await http.MultipartFile.fromPath('audio', path));

      try {
        final response = await request.send();
        if (response.statusCode == 200) {
          final data = json.decode(await response.stream.bytesToString());
          setState(() {
            resultText = data['recognized_text'] ?? 'Anlaşılamadı';
            responseText = data['response_text'] ?? '';
            status = 'Tamamlandı';
            isProcessing = false;
          });

          if (data['audio_url'] != null) {
            String audioUrl = widget.baseUrl + data['audio_url'];
            await audioPlayer.play(UrlSource(audioUrl));
          }
        } else {
          setState(() {
            status = 'Hata: ${response.statusCode}';
            isProcessing = false;
          });
        }
      } catch (e) {
        setState(() {
          status = 'Bağlantı Hatası';
          isProcessing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Çeviri Yap')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Container(
              padding: EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                children: [
                  Text('Durum: $status', style: TextStyle(color: Colors.blueAccent)),
                  SizedBox(height: 24),
                  if (isProcessing) CircularProgressIndicator(),
                  if (!isProcessing)
                    Icon(
                      isRecording ? Icons.graphic_eq : Icons.mic_none,
                      size: 80,
                      color: isRecording ? Colors.red : Colors.grey,
                    ),
                  SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: (isRecording || isProcessing) ? null : _startTranslation,
                    child: Text(isRecording ? 'KAYDEDİLİYOR...' : 'KONUŞMAK İÇİN BASIN'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                      backgroundColor: Colors.blueAccent,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 32),
            if (resultText.isNotEmpty)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Algılanan:', style: TextStyle(color: Colors.grey)),
                  Text(resultText, style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  SizedBox(height: 24),
                  if (responseText.isNotEmpty) ...[
                    Text('Yanıt:', style: TextStyle(color: Colors.grey)),
                    Text(responseText, style: TextStyle(fontSize: 20, color: Colors.greenAccent)),
                  ],
                ],
              ),
          ],
        ),
      ),
    );
  }
}
