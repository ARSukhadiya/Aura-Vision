import Foundation
import AVFoundation
import UIKit
import Vision

@MainActor
class CameraManager: NSObject, ObservableObject {
    @Published var isSessionRunning = false
    @Published var isAuthorized = false
    @Published var errorMessage: String?
    @Published var currentImageBuffer: CVPixelBuffer?
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private var videoDeviceInput: AVCaptureDeviceInput?
    
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInitiated)
    private let videoDataQueue = DispatchQueue(label: "camera.video", qos: .userInitiated)
    
    // Face detection
    private let faceDetectionRequest = VNDetectFaceLandmarksRequest()
    private var faceDetectionSequenceHandler = VNSequenceRequestHandler()
    
    // Configuration
    private let sessionPreset: AVCaptureSession.Preset = .high
    private let videoQuality: AVCaptureSession.Preset = .high
    
    override init() {
        super.init()
        setupSession()
    }
    
    func requestCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
            setupSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                    if granted {
                        self?.setupSession()
                    }
                }
            }
        case .denied, .restricted:
            isAuthorized = false
            errorMessage = "Camera access is required for emotion recognition"
        @unknown default:
            isAuthorized = false
        }
    }
    
    private func setupSession() {
        guard isAuthorized else { return }
        
        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }
    
    private func configureSession() {
        session.beginConfiguration()
        
        // Set session quality
        session.sessionPreset = sessionPreset
        
        // Add video input
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            DispatchQueue.main.async {
                self.errorMessage = "Unable to access front camera"
            }
            return
        }
        
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
            
            if session.canAddInput(videoDeviceInput) {
                session.addInput(videoDeviceInput)
                self.videoDeviceInput = videoDeviceInput
            } else {
                DispatchQueue.main.async {
                    self.errorMessage = "Unable to add video input to session"
                }
                return
            }
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Unable to create video device input: \(error.localizedDescription)"
            }
            return
        }
        
        // Add video output
        videoOutput.setSampleBufferDelegate(self, queue: videoDataQueue)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        } else {
            DispatchQueue.main.async {
                self.errorMessage = "Unable to add video output to session"
            }
            return
        }
        
        // Add photo output for still images
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
        }
        
        // Configure video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
            connection.isVideoMirrored = true
        }
        
        session.commitConfiguration()
        
        DispatchQueue.main.async {
            self.errorMessage = nil
        }
    }
    
    func startSession() {
        guard !isSessionRunning else { return }
        
        sessionQueue.async { [weak self] in
            self?.session.startRunning()
            
            DispatchQueue.main.async {
                self?.isSessionRunning = true
            }
        }
    }
    
    func stopSession() {
        guard isSessionRunning else { return }
        
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
            
            DispatchQueue.main.async {
                self?.isSessionRunning = false
            }
        }
    }
    
    func switchCamera() {
        guard isSessionRunning else { return }
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Remove current input
            if let currentInput = self.videoDeviceInput {
                self.session.removeInput(currentInput)
            }
            
            // Get new camera position
            let currentPosition = self.videoDeviceInput?.device.position ?? .front
            let newPosition: AVCaptureDevice.Position = currentPosition == .front ? .back : .front
            
            // Get new device
            guard let newDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition) else {
                return
            }
            
            // Create new input
            do {
                let newInput = try AVCaptureDeviceInput(device: newDevice)
                
                if self.session.canAddInput(newInput) {
                    self.session.addInput(newInput)
                    self.videoDeviceInput = newInput
                    
                    // Update video orientation
                    if let connection = self.videoOutput.connection(with: .video) {
                        connection.videoOrientation = .portrait
                        connection.isVideoMirrored = newPosition == .front
                    }
                }
            } catch {
                print("Error switching camera: \(error)")
            }
        }
    }
    
    func capturePhoto() {
        guard isSessionRunning else { return }
        
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    private func detectFaces(in imageBuffer: CVPixelBuffer) {
        let request = VNDetectFaceLandmarksRequest { [weak self] request, error in
            if let error = error {
                print("Face detection error: \(error)")
                return
            }
            
            guard let observations = request.results as? [VNFaceObservation] else { return }
            
            // Process face observations
            self?.processFaceObservations(observations)
        }
        
        do {
            try faceDetectionSequenceHandler.perform([request], on: imageBuffer)
        } catch {
            print("Error performing face detection: \(error)")
        }
    }
    
    private func processFaceObservations(_ observations: [VNFaceObservation]) {
        // Process detected faces
        // This could be used for additional face analysis or UI updates
        for observation in observations {
            let confidence = observation.confidence
            let boundingBox = observation.boundingBox
            
            // You could use this information for UI overlays or additional processing
            print("Face detected with confidence: \(confidence), bounds: \(boundingBox)")
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Update current image buffer
        DispatchQueue.main.async {
            self.currentImageBuffer = imageBuffer
        }
        
        // Perform face detection
        detectFaces(in: imageBuffer)
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension CameraManager: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error capturing photo: \(error)")
            return
        }
        
        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            print("Unable to create image from photo data")
            return
        }
        
        // Handle captured photo
        // You could save it, process it, or display it
        print("Photo captured successfully")
    }
}
