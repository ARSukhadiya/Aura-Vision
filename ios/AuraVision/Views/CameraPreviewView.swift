import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        
        // Create preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraManager.session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        
        view.layer.addSublayer(previewLayer)
        
        // Add face detection overlay
        let overlayView = FaceDetectionOverlayView()
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(overlayView)
        
        NSLayoutConstraint.activate([
            overlayView.topAnchor.constraint(equalTo: view.topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // Update preview layer frame when view size changes
        if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = uiView.bounds
        }
    }
}

class FaceDetectionOverlayView: UIView {
    private var faceLayers: [CAShapeLayer] = []
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .clear
    }
    
    func updateFaceDetections(_ faceObservations: [VNFaceObservation]) {
        // Remove old face layers
        faceLayers.forEach { $0.removeFromSuperlayer() }
        faceLayers.removeAll()
        
        // Add new face detection overlays
        for observation in faceObservations {
            let faceLayer = createFaceLayer(for: observation)
            layer.addSublayer(faceLayer)
            faceLayers.append(faceLayer)
        }
    }
    
    private func createFaceLayer(for observation: VNFaceObservation) -> CAShapeLayer {
        let faceLayer = CAShapeLayer()
        faceLayer.strokeColor = UIColor.green.cgColor
        faceLayer.lineWidth = 2.0
        faceLayer.fillColor = UIColor.clear.cgColor
        
        // Convert normalized coordinates to view coordinates
        let boundingBox = observation.boundingBox
        let rect = CGRect(
            x: boundingBox.origin.x * bounds.width,
            y: (1 - boundingBox.origin.y - boundingBox.height) * bounds.height,
            width: boundingBox.width * bounds.width,
            height: boundingBox.height * bounds.height
        )
        
        faceLayer.path = UIBezierPath(rect: rect).cgPath
        
        return faceLayer
    }
}

struct CameraControlsView: View {
    @ObservedObject var cameraManager: CameraManager
    
    var body: some View {
        HStack {
            // Camera switch button
            Button(action: {
                cameraManager.switchCamera()
            }) {
                Image(systemName: "camera.rotate")
                    .font(.title2)
                    .foregroundColor(.white)
                    .frame(width: 44, height: 44)
                    .background(Color.black.opacity(0.6))
                    .clipShape(Circle())
            }
            
            Spacer()
            
            // Capture photo button
            Button(action: {
                cameraManager.capturePhoto()
            }) {
                Circle()
                    .fill(Color.white)
                    .frame(width: 60, height: 60)
                    .overlay(
                        Circle()
                            .stroke(Color.black, lineWidth: 3)
                    )
            }
            
            Spacer()
            
            // Settings button
            Button(action: {
                // Handle settings
            }) {
                Image(systemName: "gear")
                    .font(.title2)
                    .foregroundColor(.white)
                    .frame(width: 44, height: 44)
                    .background(Color.black.opacity(0.6))
                    .clipShape(Circle())
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 20)
    }
}

struct CameraStatusView: View {
    @ObservedObject var cameraManager: CameraManager
    
    var body: some View {
        VStack(spacing: 10) {
            if let errorMessage = cameraManager.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.white)
                    
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
            }
            
            if !cameraManager.isAuthorized {
                HStack {
                    Image(systemName: "camera.slash.fill")
                        .foregroundColor(.red)
                    
                    Text("Camera access required")
                        .font(.caption)
                        .foregroundColor(.white)
                    
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
            }
            
            if cameraManager.isSessionRunning {
                HStack {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                    
                    Text("Recording")
                        .font(.caption)
                        .foregroundColor(.white)
                    
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
            }
        }
        .padding(.top, 10)
    }
}

struct CameraPreviewContainer: View {
    @ObservedObject var cameraManager: CameraManager
    
    var body: some View {
        ZStack {
            // Camera preview
            CameraPreviewView(cameraManager: cameraManager)
                .cornerRadius(20)
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                )
            
            // Status overlay
            VStack {
                CameraStatusView(cameraManager: cameraManager)
                
                Spacer()
                
                // Camera controls
                CameraControlsView(cameraManager: cameraManager)
            }
        }
    }
}

#Preview {
    CameraPreviewContainer(cameraManager: CameraManager())
}
