import SwiftUI

struct EmotionDisplayView: View {
    @ObservedObject var emotionManager: EmotionRecognitionManager
    
    var body: some View {
        VStack(spacing: 15) {
            // Main emotion display
            VStack {
                // Large emoji
                Text(emotionManager.currentEmotion.emoji)
                    .font(.system(size: 60))
                    .scaleEffect(emotionManager.isProcessing ? 1.1 : 1.0)
                    .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: emotionManager.isProcessing)
                
                // Emotion name
                Text(emotionManager.currentEmotion.rawValue.capitalized)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                // Confidence bar
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Confidence")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text("\(Int(emotionManager.emotionConfidence * 100))%")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                    }
                    
                    // Progress bar
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            // Background
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.gray.opacity(0.3))
                                .frame(height: 8)
                            
                            // Progress
                            RoundedRectangle(cornerRadius: 4)
                                .fill(confidenceColor)
                                .frame(width: geometry.size.width * CGFloat(emotionManager.emotionConfidence), height: 8)
                                .animation(.easeInOut(duration: 0.3), value: emotionManager.emotionConfidence)
                        }
                    }
                    .frame(height: 8)
                }
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 15)
                    .fill(Color(.systemBackground))
                    .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
            )
            
            // Emotion statistics
            if !emotionManager.emotionHistory.isEmpty {
                EmotionStatisticsView(emotionManager: emotionManager)
            }
        }
    }
    
    private var confidenceColor: Color {
        let confidence = emotionManager.emotionConfidence
        
        switch confidence {
        case 0.8...:
            return .green
        case 0.6..<0.8:
            return .orange
        default:
            return .red
        }
    }
}

struct EmotionStatisticsView: View {
    @ObservedObject var emotionManager: EmotionRecognitionManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recent Emotions")
                .font(.headline)
                .foregroundColor(.primary)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 10) {
                ForEach(Emotion.allCases, id: \.self) { emotion in
                    EmotionStatCard(
                        emotion: emotion,
                        count: emotionCount(for: emotion),
                        total: emotionManager.emotionHistory.count
                    )
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
    
    private func emotionCount(for emotion: Emotion) -> Int {
        emotionManager.emotionHistory.filter { $0.emotion == emotion }.count
    }
}

struct EmotionStatCard: View {
    let emotion: Emotion
    let count: Int
    let total: Int
    
    var body: some View {
        VStack(spacing: 5) {
            Text(emotion.emoji)
                .font(.title2)
            
            Text("\(count)")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
            
            if total > 0 {
                Text("\(Int(Double(count) / Double(total) * 100))%")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6))
        )
    }
}

struct EmotionTrendView: View {
    @ObservedObject var emotionManager: EmotionRecognitionManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Emotion Trend")
                .font(.headline)
                .foregroundColor(.primary)
            
            HStack {
                Text("Current Trend:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(emotionManager.getEmotionTrend().emoji)
                    .font(.title2)
                
                Text(emotionManager.getEmotionTrend().rawValue.capitalized)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
            }
            
            // Simple trend visualization
            if emotionManager.emotionHistory.count >= 5 {
                EmotionTrendChart(emotionHistory: Array(emotionManager.emotionHistory.suffix(10)))
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

struct EmotionTrendChart: View {
    let emotionHistory: [EmotionResult]
    
    var body: some View {
        HStack(spacing: 2) {
            ForEach(Array(emotionHistory.enumerated()), id: \.offset) { index, result in
                VStack {
                    Text(result.emotion.emoji)
                        .font(.caption)
                    
                    Rectangle()
                        .fill(emotionColor(for: result.emotion))
                        .frame(height: CGFloat(result.confidence * 20))
                        .cornerRadius(2)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .frame(height: 30)
    }
    
    private func emotionColor(for emotion: Emotion) -> Color {
        switch emotion {
        case .happy: return .yellow
        case .sad: return .blue
        case .angry: return .red
        case .surprise: return .orange
        case .fear: return .purple
        case .disgust: return .brown
        case .neutral: return .gray
        }
    }
}

#Preview {
    EmotionDisplayView(emotionManager: EmotionRecognitionManager())
}
