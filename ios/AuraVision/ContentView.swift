import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var emotionManager = EmotionRecognitionManager()
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var audioManager = AudioManager()
    
    @State private var isRecording = false
    @State private var showSettings = false
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.blue.opacity(0.3),
                        Color.purple.opacity(0.3)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 20) {
                    // Header
                    VStack {
                        Text("Aura-Vision")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)
                        
                        Text("Multimodal Emotion Recognition")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top)
                    
                    // Camera preview
                    CameraPreviewView(cameraManager: cameraManager)
                        .frame(height: 300)
                        .cornerRadius(20)
                        .shadow(radius: 10)
                        .padding(.horizontal)
                    
                    // Emotion display
                    EmotionDisplayView(emotionManager: emotionManager)
                        .frame(height: 120)
                        .padding(.horizontal)
                    
                    // Transcription view
                    TranscriptionView(audioManager: audioManager)
                        .frame(height: 100)
                        .padding(.horizontal)
                    
                    // Control buttons
                    HStack(spacing: 30) {
                        // Start/Stop button
                        Button(action: {
                            if isRecording {
                                stopRecording()
                            } else {
                                startRecording()
                            }
                        }) {
                            VStack {
                                Image(systemName: isRecording ? "stop.circle.fill" : "play.circle.fill")
                                    .font(.system(size: 50))
                                    .foregroundColor(isRecording ? .red : .green)
                                
                                Text(isRecording ? "Stop" : "Start")
                                    .font(.caption)
                                    .foregroundColor(.primary)
                            }
                        }
                        
                        // Settings button
                        Button(action: {
                            showSettings = true
                        }) {
                            VStack {
                                Image(systemName: "gear")
                                    .font(.system(size: 30))
                                    .foregroundColor(.blue)
                                
                                Text("Settings")
                                    .font(.caption)
                                    .foregroundColor(.primary)
                            }
                        }
                    }
                    .padding(.top, 20)
                    
                    Spacer()
                }
            }
        }
        .onAppear {
            setupManagers()
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
    }
    
    private func setupManagers() {
        // Initialize camera and audio permissions
        cameraManager.requestCameraPermission()
        audioManager.requestMicrophonePermission()
        
        // Setup emotion recognition
        emotionManager.setup()
    }
    
    private func startRecording() {
        isRecording = true
        
        // Start camera and audio capture
        cameraManager.startSession()
        audioManager.startRecording()
        
        // Start emotion recognition
        emotionManager.startRecognition(
            cameraManager: cameraManager,
            audioManager: audioManager
        )
    }
    
    private func stopRecording() {
        isRecording = false
        
        // Stop all capture and recognition
        cameraManager.stopSession()
        audioManager.stopRecording()
        emotionManager.stopRecognition()
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section("Camera Settings") {
                    HStack {
                        Text("Camera Quality")
                        Spacer()
                        Text("High")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Frame Rate")
                        Spacer()
                        Text("30 FPS")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Audio Settings") {
                    HStack {
                        Text("Sample Rate")
                        Spacer()
                        Text("16 kHz")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Audio Quality")
                        Spacer()
                        Text("High")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Emotion Recognition") {
                    HStack {
                        Text("Confidence Threshold")
                        Spacer()
                        Text("0.7")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Update Frequency")
                        Spacer()
                        Text("1 Hz")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Privacy") {
                    HStack {
                        Text("Data Processing")
                        Spacer()
                        Text("On-Device")
                            .foregroundColor(.green)
                    }
                    
                    HStack {
                        Text("Data Storage")
                        Spacer()
                        Text("None")
                            .foregroundColor(.green)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
