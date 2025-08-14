import Foundation
import AVFoundation
import Speech
import Combine

@MainActor
class AudioManager: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var isAuthorized = false
    @Published var errorMessage: String?
    @Published var currentTranscription = ""
    @Published var currentAudioData: Data?
    
    private let audioEngine = AVAudioEngine()
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    
    private let audioQueue = DispatchQueue(label: "audio.processing", qos: .userInitiated)
    private let audioBufferSize: AVAudioFrameCount = 1024
    
    // Audio configuration
    private let sampleRate: Double = 16000
    private let numberOfChannels: AVAudioChannelCount = 1
    private let audioFormat: AVAudioFormat
    
    // Audio buffer for processing
    private var audioBuffer = Data()
    private let maxBufferSize = 16000 * 3 // 3 seconds of audio at 16kHz
    
    override init() {
        // Initialize audio format
        audioFormat = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: numberOfChannels
        )!
        
        super.init()
        setupAudioSession()
    }
    
    func requestMicrophonePermission() {
        switch AVAudioSession.sharedInstance().recordPermission {
        case .granted:
            isAuthorized = true
            requestSpeechRecognitionPermission()
        case .denied:
            isAuthorized = false
            errorMessage = "Microphone access is required for speech recognition"
        case .undetermined:
            AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                    if granted {
                        self?.requestSpeechRecognitionPermission()
                    } else {
                        self?.errorMessage = "Microphone access is required for speech recognition"
                    }
                }
            }
        @unknown default:
            isAuthorized = false
        }
    }
    
    private func requestSpeechRecognitionPermission() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                switch status {
                case .authorized:
                    self?.setupSpeechRecognition()
                case .denied, .restricted, .notDetermined:
                    self?.errorMessage = "Speech recognition permission is required"
                @unknown default:
                    self?.errorMessage = "Speech recognition permission is required"
                }
            }
        }
    }
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to setup audio session: \(error)")
            errorMessage = "Failed to setup audio session"
        }
    }
    
    private func setupSpeechRecognition() {
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            errorMessage = "Speech recognition is not available"
            return
        }
        
        print("Speech recognition setup complete")
    }
    
    func startRecording() {
        guard isAuthorized else {
            errorMessage = "Microphone permission not granted"
            return
        }
        
        guard !isRecording else { return }
        
        audioQueue.async { [weak self] in
            self?.startAudioCapture()
        }
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        audioQueue.async { [weak self] in
            self?.stopAudioCapture()
        }
    }
    
    private func startAudioCapture() {
        // Reset audio buffer
        audioBuffer.removeAll()
        
        // Setup audio input
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        // Install tap on input node
        inputNode.installTap(onBus: 0, bufferSize: audioBufferSize, format: recordingFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }
        
        // Start audio engine
        do {
            audioEngine.prepare()
            try audioEngine.start()
            
            DispatchQueue.main.async {
                self.isRecording = true
                self.errorMessage = nil
            }
            
            // Start speech recognition
            startSpeechRecognition()
            
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Failed to start audio recording: \(error.localizedDescription)"
            }
        }
    }
    
    private func stopAudioCapture() {
        // Stop speech recognition
        stopSpeechRecognition()
        
        // Stop audio engine
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        DispatchQueue.main.async {
            self.isRecording = false
        }
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        
        let frameLength = Int(buffer.frameLength)
        let audioData = Data(bytes: channelData, count: frameLength * MemoryLayout<Float>.size)
        
        // Add to buffer
        audioBuffer.append(audioData)
        
        // Maintain buffer size
        if audioBuffer.count > maxBufferSize {
            audioBuffer.removeFirst(audioBuffer.count - maxBufferSize)
        }
        
        // Update current audio data
        DispatchQueue.main.async {
            self.currentAudioData = self.audioBuffer
        }
    }
    
    private func startSpeechRecognition() {
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else { return }
        
        // Cancel any existing recognition task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create new recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        
        recognitionRequest.shouldReportPartialResults = true
        
        // Start recognition
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Speech recognition error: \(error)")
                return
            }
            
            if let result = result {
                DispatchQueue.main.async {
                    self.currentTranscription = result.bestTranscription.formattedString
                }
            }
        }
        
        // Install tap for speech recognition
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: audioBufferSize, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
    }
    
    private func stopSpeechRecognition() {
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
    }
    
    func clearTranscription() {
        currentTranscription = ""
    }
    
    func getAudioFeatures() -> [Float] {
        // Extract audio features from current buffer
        // This would implement MFCC, spectral features, etc.
        // For now, return a placeholder
        
        guard let audioData = currentAudioData else { return [] }
        
        // Convert Data to Float array
        let floatCount = audioData.count / MemoryLayout<Float>.size
        let floatArray = audioData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self).prefix(floatCount))
        }
        
        // Simple feature extraction (RMS energy)
        let rms = sqrt(floatArray.map { $0 * $0 }.reduce(0, +) / Float(floatArray.count))
        
        return [rms]
    }
    
    func getAudioSpectrum() -> [Float] {
        // Return frequency spectrum for visualization
        // This would implement FFT or similar
        guard let audioData = currentAudioData else { return [] }
        
        let floatCount = audioData.count / MemoryLayout<Float>.size
        let floatArray = audioData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self).prefix(floatCount))
        }
        
        // Simple spectrum simulation
        var spectrum: [Float] = []
        for i in 0..<64 {
            let frequency = Float(i) / 64.0
            let amplitude = sin(frequency * Float.pi * 2) * 0.5 + 0.5
            spectrum.append(amplitude)
        }
        
        return spectrum
    }
    
    func exportAudioData() -> Data? {
        // Export current audio buffer
        return currentAudioData
    }
    
    func getTranscriptionHistory() -> [String] {
        // Return recent transcriptions
        // This could be expanded to store a history of transcriptions
        return currentTranscription.isEmpty ? [] : [currentTranscription]
    }
}
