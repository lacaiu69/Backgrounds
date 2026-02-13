import SwiftUI
import MetalKit

enum BackgroundMood: Int, CaseIterable {
    case soothing = 0
    case neutral = 1
    case energetic = 2

    var title: String {
        switch self {
        case .soothing: return "Soothing"
        case .neutral: return "Neutral"
        case .energetic: return "Energetic"
        }
    }
}

import simd

struct AdaptiveBackgroundUniforms {
    var time: Float
    var pad0: Float = 0
    var resolution: SIMD2<Float>

    var fromMood: Float
    var toMood: Float
    var transition: Float
    var pad1: Float = 0
}



final class MetalBackgroundRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState!
    private var startTime = CACurrentMediaTime()
    var targetMoodIndex: Float = 1.0 // set by buttons/data later

    private var fromMoodIndex: Float = 1.0
    private var toMoodIndex: Float = 1.0
    private var transitionT: Float = 1.0

    private var lastTime = CACurrentMediaTime()
    private var currentMoodVel: Float = 0   // for spring (optional)

    // Target mood is set from SwiftUI (buttons now, data later)
    var targetMood: Float = 1.0
    private var currentMood: Float = 1.0

    init?(mtkView: MTKView) {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let cq = dev.makeCommandQueue()
        else { return nil }

        device = dev
        commandQueue = cq
        super.init()

        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false
        mtkView.preferredFramesPerSecond = 60
        mtkView.clearColor = MTLClearColorMake(1, 1, 1, 1)

        do {
            let library = try device.makeDefaultLibrary(bundle: .main)
            guard let vertex = library.makeFunction(name: "vertex_main"),
                  let fragment = library.makeFunction(name: "fragment_main")
            else { return nil }

            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vertex
            desc.fragmentFunction = fragment
            desc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat

            pipelineState = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            print("Pipeline error:", error)
            return nil
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd)
        else { return }

        let now = CACurrentMediaTime()
        let dt = Float(min(now - lastTime, 1.0/30.0))
        lastTime = now

        // If target changed, start a new crossfade
        if toMoodIndex != targetMoodIndex {
            // Start new transition from what is ACTUALLY on screen right now
            let displayed = fromMoodIndex + (toMoodIndex - fromMoodIndex) * transitionT
            fromMoodIndex = displayed
            toMoodIndex = targetMoodIndex
            transitionT = 0.0
        }


        // Advance transition (duration in seconds)
        let duration: Float = 1.0   // try 0.8..1.6
        transitionT = min(transitionT + dt / duration, 1.0)

        // Critically damped spring toward targetMood
        let omega: Float = 10.0  // higher = snappier, lower = floatier (try 8..14)
        let zeta:  Float = 1.0   // 1.0 = critically damped (no overshoot)

        let x = currentMood
        let v = currentMoodVel
        let goal = targetMood

        let f = 1.0 + 2.0 * dt * zeta * omega
        let oo = omega * omega
        let hoo = dt * oo
        let hhoo = dt * hoo
        let detInv = 1.0 / (f + hhoo)

        let xNew = (f * x + dt * v + hhoo * goal) * detInv
        let vNew = (v + hoo * (goal - x)) * detInv

        currentMood = xNew
        currentMoodVel = vNew


        let t = Float(CACurrentMediaTime() - startTime)
        let res = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))

        var u = AdaptiveBackgroundUniforms(
            time: t,
            pad0: 0,
            resolution: res,
            fromMood: fromMoodIndex,
            toMood: toMoodIndex,
            transition: transitionT,
            pad1: 0
        )

        enc.setRenderPipelineState(pipelineState)
        enc.setFragmentBytes(&u, length: MemoryLayout<AdaptiveBackgroundUniforms>.stride, index: 0)

        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}

struct MetalBackgroundMTKView: UIViewRepresentable {
    var mood: BackgroundMood

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.backgroundColor = .clear

        guard let renderer = MetalBackgroundRenderer(mtkView: view) else { return view }

        context.coordinator.renderer = renderer
        view.delegate = renderer

        renderer.targetMoodIndex = Float(mood.rawValue)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer?.targetMoodIndex = Float(mood.rawValue)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var renderer: MetalBackgroundRenderer? }
}
