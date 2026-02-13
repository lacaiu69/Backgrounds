import SwiftUI

struct ContentView: View {
    @State private var debugMood: BackgroundMood = .neutral

    var body: some View {
        ZStack {
            MetalBackgroundMTKView(mood: debugMood)
                .ignoresSafeArea()

            // Debug-only buttons (remove in production)
            VStack {
                Spacer()
                HStack(spacing: 10) {
                    ForEach(BackgroundMood.allCases, id: \.self) { m in
                        Button(m.title) { debugMood = m }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(.white.opacity(0.75))
                            .clipShape(Capsule())
                    }
                }
                .padding(.bottom, 28)
            }
        }
    }
}


#Preview {
    ContentView()
}
