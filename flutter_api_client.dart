
// bus_stop_service.dart
// Flutter service to call prediction API

import 'dart:convert';
import 'package:http/http.dart' as http;

class BusStopPrediction {
  final List<double> predictions;
  final List<String> timestamps;
  final String generatedAt;
  final String location;

  BusStopPrediction({
    required this.predictions,
    required this.timestamps,
    required this.generatedAt,
    required this.location,
  });

  factory BusStopPrediction.fromJson(Map<String, dynamic> json) {
    return BusStopPrediction(
      predictions: List<double>.from(json['predictions'].map((x) => x.toDouble())),
      timestamps: List<String>.from(json['timestamps']),
      generatedAt: json['generated_at'],
      location: json['location'],
    );
  }
}

class BusStopApiService {
  static const String baseUrl = 'http://localhost:8000'; // Change to your server IP
  
  // Get predictions for next 6 hours
  static Future<BusStopPrediction> getPredictions(
    List<double> historicalData,
    String location,
  ) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'historical_data': historicalData,
          'location': location,
        }),
      );

      if (response.statusCode == 200) {
        return BusStopPrediction.fromJson(json.decode(response.body));
      } else {
        throw Exception('Failed to get predictions: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('API call failed: $e');
    }
  }

  // Check if API is running
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

// Example usage in Flutter widget
/*
class PredictionWidget extends StatefulWidget {
  @override
  _PredictionWidgetState createState() => _PredictionWidgetState();
}

class _PredictionWidgetState extends State<PredictionWidget> {
  List<double> historicalData = []; // Get from your MongoDB/state
  BusStopPrediction? prediction;
  bool loading = false;

  Future<void> fetchPrediction() async {
    setState(() => loading = true);
    try {
      prediction = await BusStopApiService.getPredictions(
        historicalData,
        'bus_stop_1',
      );
    } catch (e) {
      print('Error: $e');
    }
    setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ElevatedButton(
          onPressed: fetchPrediction,
          child: Text('Get Predictions'),
        ),
        if (loading) CircularProgressIndicator(),
        if (prediction != null)
          Column(
            children: [
              Text('Next 6 hours prediction:'),
              ...prediction!.predictions.asMap().entries.map((entry) {
                int index = entry.key;
                double count = entry.value;
                return ListTile(
                  title: Text('${prediction!.timestamps[index]}'),
                  trailing: Text('${count.toStringAsFixed(1)} people'),
                );
              }).toList(),
            ],
          ),
      ],
    );
  }
}
*/
