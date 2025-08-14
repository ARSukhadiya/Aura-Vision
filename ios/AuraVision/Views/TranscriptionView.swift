import SwiftUI

struct TranscriptionView: View {
    @ObservedObject var audioManager: AudioManager
    
    var body: some View {
        VStack(spacing: 15) {
            // Header
            HStack {
                Image(systemName: "mic.fill")
                    .foregroundColor(.blue)
                
                Text("Speech Recognition")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                // Recording indicator
                if audioManager.isRecording {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                            .scaleEffect(audioManager.isRecording ? 1.2 : 1.0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: audioManager.isRecording)
                        
                        Text("Recording")
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
            }
            
            // Transcription display
            VStack(alignment: .leading, spacing: 10) {
                if audioManager.currentTranscription.isEmpty {
                    // Placeholder
                    HStack {
                        Image(systemName: "text.bubble")
                            .foregroundColor(.gray)
                        
                        Text("Start speaking to see transcription...")
                            .font(.body)
                            .foregroundColor(.gray)
                            .italic()
                        
                        Spacer()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                } else {
                    // Actual transcription
                    ScrollView {
                        Text(audioManager.currentTranscription)
                            .font(.body)
                            .foregroundColor(.primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color(.systemBackground))
                            .cornerRadius(10)
                            .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
                    }
                    .frame(maxHeight: 80)
                }
            }
            
            // Audio visualization
            AudioVisualizationView(audioManager: audioManager)
            
            // Controls
            HStack {
                // Clear button
                Button(action: {
                    audioManager.clearTranscription()
                }) {
                    HStack {
                        Image(systemName: "trash")
                        Text("Clear")
                    }
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
                }
                
                Spacer()
                
                // Word count
                Text("\(wordCount) words")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
    
    private var wordCount: Int {
        audioManager.currentTranscription.split(separator: " ").count
    }
}

struct AudioVisualizationView: View {
    @ObservedObject var audioManager: AudioManager
    @State private var spectrum: [Float] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Audio Level")
                .font(.caption)
                .foregroundColor(.secondary)
            
            // Audio spectrum visualization
            HStack(spacing: 2) {
                ForEach(Array(spectrum.enumerated()), id: \.offset) { index, amplitude in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(audioColor(for: amplitude))
                        .frame(width: 3, height: CGFloat(amplitude * 30))
                        .animation(.easeInOut(duration: 0.1), value: amplitude)
                }
            }
            .frame(height: 30)
            .onReceive(Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()) { _ in
                if audioManager.isRecording {
                    spectrum = audioManager.getAudioSpectrum()
                } else {
                    spectrum = Array(repeating: 0, count: 64)
                }
            }
        }
    }
    
    private func audioColor(for amplitude: Float) -> Color {
        if amplitude > 0.7 {
            return .red
        } else if amplitude > 0.4 {
            return .orange
        } else if amplitude > 0.1 {
            return .green
        } else {
            return .gray.opacity(0.3)
        }
    }
}

struct TranscriptionHistoryView: View {
    @ObservedObject var audioManager: AudioManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Transcription History")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Button("Clear All") {
                    // Clear all transcriptions
                }
                .font(.caption)
                .foregroundColor(.red)
            }
            
            if audioManager.getTranscriptionHistory().isEmpty {
                Text("No transcriptions yet")
                    .font(.body)
                    .foregroundColor(.gray)
                    .italic()
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(Array(audioManager.getTranscriptionHistory().enumerated()), id: \.offset) { index, transcription in
                            TranscriptionHistoryItem(
                                transcription: transcription,
                                index: index
                            )
                        }
                    }
                }
                .frame(maxHeight: 200)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

struct TranscriptionHistoryItem: View {
    let transcription: String
    let index: Int
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("#\(index + 1)")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.blue)
                
                Spacer()
                
                Text("\(transcription.split(separator: " ").count) words")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(transcription)
                .font(.body)
                .foregroundColor(.primary)
                .lineLimit(3)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

struct SpeechRecognitionStatusView: View {
    @ObservedObject var audioManager: AudioManager
    
    var body: some View {
        HStack(spacing: 15) {
            // Microphone status
            VStack {
                Image(systemName: audioManager.isAuthorized ? "mic.fill" : "mic.slash.fill")
                    .font(.title2)
                    .foregroundColor(audioManager.isAuthorized ? .green : .red)
                
                Text(audioManager.isAuthorized ? "Ready" : "No Access")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Recording status
            VStack {
                Image(systemName: audioManager.isRecording ? "record.circle.fill" : "record.circle")
                    .font(.title2)
                    .foregroundColor(audioManager.isRecording ? .red : .gray)
                
                Text(audioManager.isRecording ? "Recording" : "Stopped")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Error status
            if let errorMessage = audioManager.errorMessage {
                VStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title2)
                        .foregroundColor(.orange)
                    
                    Text("Error")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.systemGray6))
        )
    }
}

#Preview {
    VStack(spacing: 20) {
        TranscriptionView(audioManager: AudioManager())
        TranscriptionHistoryView(audioManager: AudioManager())
        SpeechRecognitionStatusView(audioManager: AudioManager())
    }
    .padding()
}
