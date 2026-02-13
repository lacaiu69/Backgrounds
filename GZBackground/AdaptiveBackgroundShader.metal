#include <metal_stdlib>
using namespace metal;

// ============================================================
//  Adaptive Background — 3 moods (Soothing / Neutral / Energetic)
//  - Smooth crossfade between fromMood -> toMood (no stepping through neutral)
//  - Each mood has a distinct "hero motion" that's visible
//  - Energetic hero includes cool-core -> warm-edge temperature shift
// ============================================================

struct AdaptiveBackgroundUniforms {
    float time;
    float _pad0;
    float2 resolution;
    float fromMood;     // 0 = Soothing, 1 = Neutral, 2 = Energetic
    float toMood;       // 0 = Soothing, 1 = Neutral, 2 = Energetic
    float transition;   // 0..1
    float _pad1;
};

struct VertexOut { float4 position [[position]]; };

// Fullscreen quad (2 triangles)
vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    float2 pos[6] = {
        {-1,-1},{ 1,-1},{-1, 1},
        {-1, 1},{ 1,-1},{ 1, 1}
    };
    VertexOut o;
    o.position = float4(pos[vid], 0, 1);
    return o;
}

// ============================================================
// Utilities
// ============================================================

static inline float sCurve(float x) { return x * x * (3.0 - 2.0 * x); }

static inline float3 desat(float3 c, float keep) {
    float l = dot(c, float3(0.299, 0.587, 0.114));
    return mix(float3(l), c, keep);
}

// Screen blend, controlled by alpha a
static inline float3 screenBlend(float3 base, float3 top, float a) {
    float3 s = 1.0 - (1.0 - base) * (1.0 - top);
    return mix(base, s, a);
}

// ============================================================
// Noise (value noise + fbm)
// ============================================================

static inline float hash21(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453123);
}

static inline float noise21(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);

    float a = hash21(i);
    float b = hash21(i + float2(1, 0));
    float c = hash21(i + float2(0, 1));
    float d = hash21(i + float2(1, 1));

    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

static inline float fbm4(float2 p) {
    float f = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        f += a * noise21(p);
        p = p * 2.02 + float2(11.3, 7.7);
        a *= 0.5;
    }
    return f;
}

// ============================================================
// Palette
// ============================================================

struct Palette { float3 A; float3 B; float3 C; };

// warmth: 0 (cool) -> 1 (warm)
static inline Palette paletteForWarmth(float warmth) {
    // cool
    const float3 coolA = float3(0.576, 0.584, 0.776); // #9395C6
    const float3 coolB = float3(0.514, 0.584, 0.671); // #8395AB
    const float3 coolC = float3(0.769, 0.808, 0.769); // #C4CEC4

    // neutral
    const float3 neuA  = float3(0.925, 0.894, 0.882); // #EFE4E1
    const float3 neuB  = float3(0.882, 0.816, 0.796); // #E1D0CB
    const float3 neuC  = float3(0.855, 0.812, 0.765); // #DACFC3

    // warm
    const float3 warmA = float3(0.941, 0.871, 0.819);
    const float3 warmB = float3(0.816, 0.620, 0.624); // #D09E9F
    const float3 warmC = float3(0.604, 0.569, 0.522); // #9A9185

    float coolToNeu = sCurve(smoothstep(0.0, 0.5, warmth));
    float neuToWarm = sCurve(smoothstep(0.5, 1.0, warmth));
    float mid       = sCurve(smoothstep(0.45, 0.55, warmth));

    float3 A_cn = mix(coolA, neuA, coolToNeu);
    float3 B_cn = mix(coolB, neuB, coolToNeu);
    float3 C_cn = mix(coolC, neuC, coolToNeu);

    float3 A_nw = mix(neuA, warmA, neuToWarm);
    float3 B_nw = mix(neuB, warmB, neuToWarm);
    float3 C_nw = mix(neuC, warmC, neuToWarm);

    Palette p;
    p.A = mix(A_cn, A_nw, mid);
    p.B = mix(B_cn, B_nw, mid);
    p.C = mix(C_cn, C_nw, mid);
    return p;
}

// ============================================================
// Hero Motions
// ============================================================

// Soothing: interference bands
static inline float heroSoothing(float2 q, float t) {
    float ph = t * 0.35;
    float a = sin(q.x * 3.0 + ph);
    float b = sin(q.y * 3.6 - ph * 1.05);
    float c = sin((q.x + q.y) * 2.6 + ph * 0.8);

    float inter = (a + b + 0.8 * c) / 2.8;  // [-1..1]
    inter = 0.5 + 0.5 * inter;              // [0..1]
    inter = smoothstep(0.18, 0.92, inter);   // contrast
    return inter;
}

// Neutral: glassy sweep + micro texture
static inline float heroNeutral(float2 q, float t) {
    float2 p = float2(q.x * 0.80 - q.y * 0.60,
                      q.x * 0.60 + q.y * 0.80);

    float cx = -0.55 + fract(t * 0.07) * 1.10;
    float d  = abs(p.x - cx);

    float core = exp(-(d * d) / (0.08 * 0.08));
    float tail = exp(-(d * d) / (0.18 * 0.18)) * 0.55;

    float micro = fbm4(q * 2.6 + float2(0.0, t * 0.08));
    micro = smoothstep(0.45, 0.92, micro) * 0.30;

    return clamp(core + tail + micro, 0.0, 1.0);
}
// Returns a signed cloth "height" (≈ displacement) and rN for tint mapping.
// This is NOT a brightness mask — it's used to compute a pseudo normal.
static inline float2 energeticHeightField(float2 q, float t) {
    const float ex = 1.18;
    const float ey = 0.92;

    float2 c = float2(-0.78, 0.62);
    c += float2(0.03 * sin(t * 0.22), 0.03 * cos(t * 0.18));

    float2 p = q - c;
    float2 e = float2(p.x / ex, p.y / ey);

    float r  = length(e);
    float rN = clamp(r / 1.55, 0.0, 1.0);

    float wob = fbm4(e * 1.9 + float2(t * 0.18, -t * 0.12));
    wob = (wob - 0.5) * 0.22;

    float speed = 1.05 + 0.18 * sin(t * 0.22) + 0.10 * (fbm4(float2(t*0.07, 3.1)) - 0.5);

    float phase = r * 9.2 - t * (speed * 4.6);
    float sRaw  = sin(phase + wob * 2.4);     // [-1..1] signed wave

    // Signed "cloth displacement" (keep subtle, this drives normals)
    float height = sRaw * 0.085;              // try 0.06..0.12

    // Fade displacement near origin + far out (keeps it believable)
    float nearFade = smoothstep(0.20, 0.55, r);
    float farFade  = 1.0 - smoothstep(1.15, 1.65, r);
    height *= nearFade * farFade;

    return float2(height, rN);
}

static inline float2 heroEnergeticField(float2 q, float t) {
    const float ex = 1.18;
    const float ey = 0.92;

    float2 c = float2(-0.78, 0.62);
    c += float2(0.03 * sin(t * 0.22), 0.03 * cos(t * 0.18));

    float2 p = q - c;
    float2 e = float2(p.x / ex, p.y / ey);

    float r  = length(e);
    float rN = clamp(r / 1.55, 0.0, 1.0);

    float wob = fbm4(e * 1.9 + float2(t * 0.18, -t * 0.12));
    wob = (wob - 0.5) * 0.22;

    float speed = 1.05 + 0.18 * sin(t * 0.22) + 0.10 * (fbm4(float2(t*0.07, 3.1)) - 0.5);
    float phase = r * 9.2 - t * (speed * 4.6);

    float s = 0.5 + 0.5 * sin(phase + wob * 2.4); // [0..1]

    // Crest mask (NOT white fill) — wider/clothy
    float crests = smoothstep(0.62, 0.92, s);
    float trough = smoothstep(0.50, 0.88, 1.0 - s);
    float hero = mix(crests, trough, 0.28);

    // Intermittency (gentle gates)
    float envBase = 0.60 + 0.40 * sin(t * 0.12 + 1.2);
    envBase = smoothstep(0.25, 0.90, envBase);

    float envIrreg = fbm4(float2(t * 0.045, 9.7));
    envIrreg = smoothstep(0.40, 0.85, envIrreg);

    float envelope = mix(0.55, 1.0, envBase * 0.6 + envIrreg * 0.4);

    float packetPhase = r * 1.45 - t * 0.45 + wob * 0.6;
    float packet = 0.5 + 0.5 * sin(packetPhase);
    packet = smoothstep(0.30, 0.85, packet);

    float patch = fbm4(e * 1.05 + float2(t * 0.05, -t * 0.03));
    patch = smoothstep(0.40, 0.85, patch);

    hero *= envelope;
    hero *= mix(0.65, 1.0, packet);
    hero *= mix(0.75, 1.0, patch);

    float nearFade = smoothstep(0.20, 0.55, r);
    float farFade  = 1.0 - smoothstep(1.15, 1.65, r);
    hero *= nearFade * farFade;

    // Soft contrast, but keep readable
    hero = hero * (0.75 + 0.55 * hero);
    hero = clamp(hero, 0.0, 1.0);

    return float2(hero, rN);
}



// ============================================================
// Render a single mood
// ============================================================

static inline float3 renderOne(float2 uv, float2 uv01, float2 res, float t, float mood) {
    // Mood parameters
    float energy = (mood < 0.5) ? 0.22 : ((mood < 1.5) ? 0.40 : 0.72);
    float warmth = (mood < 0.5) ? 0.22 : ((mood < 1.5) ? 0.50 : 0.85);

    Palette pal = paletteForWarmth(warmth);

    // Base motion time (kept calm so hero reads)
    float tt = t * mix(0.035, 0.11, sCurve(energy));

    // Base warp
    float2 warp = float2(
        noise21(uv * 1.10 + float2(0.0,  tt * 0.35)),
        noise21(uv * 1.10 + float2(2.1, -tt * 0.32))
    );
    warp = (warp - 0.5) * mix(0.028, 0.055, sCurve(energy));

    float2 q = uv + warp;

    // Base gradient mix
    float g1 = smoothstep(-0.62, 0.62, q.y + 0.08 * sin(tt * 0.60));
    float g2 = smoothstep(-0.82, 0.82, q.x * 0.60 + q.y * 0.20);
    float m  = 0.60 * g1 + 0.40 * g2;

    float3 col = (m < 0.5) ? mix(pal.A, pal.B, m * 2.0)
                           : mix(pal.B, pal.C, (m - 0.5) * 2.0);

    // Subtle base bloom
    {
        float2 c = float2(0.18 * sin(tt * 0.55), 0.16 * cos(tt * 0.48));
        float d = length(q - c);
        float bloom = exp(-(d * d) / (0.36 * 0.36));
        col = mix(col, mix(pal.B, pal.A, 0.45), bloom * 0.07);
    }

    // ---------------- HERO (state unique) ----------------
    float hero = 0.0;
    float3 heroTint = pal.B;

    if (mood < 0.5) {
        // Soothing
        hero = heroSoothing(q * 1.05, tt);
        heroTint = mix(pal.A, pal.B, 0.70);

        hero = clamp(pow(hero, 0.75) * 1.15, 0.0, 1.0);

    } else if (mood < 1.5) {
        // Neutral
        hero = heroNeutral(q, tt);
        heroTint = mix(pal.B, pal.A, 0.55);

        hero = clamp(pow(hero, 0.90) * 1.00, 0.0, 1.0);

    } else {
        // Energetic: cloth depth lighting (NOT a light beam)
        float2 hf = heroEnergeticField(q * 1.10, tt * 2.8);
        hero = hf.x;

        float2 hh = energeticHeightField(q * 1.10, tt * 2.8);
        float height = hh.x;
        float rN = hf.y;

        // --- compute pseudo-normal from height (finite differences) ---
        float2 du = float2(2.0 / res.x, 0.0);
        float2 dv = float2(0.0, 2.0 / res.y);

        float hU = energeticHeightField((q + du) * 1.10, tt * 2.8).x;
        float hV = energeticHeightField((q + dv) * 1.10, tt * 2.8).x;

        float dhdx = (hU - height) / max(du.x, 1e-6);
        float dhdy = (hV - height) / max(dv.y, 1e-6);

        // Strength of depth (cloth bump)
        float bump = 0.85; // try 0.6..1.2
        float3 N = normalize(float3(-dhdx * bump, -dhdy * bump, 1.0));

        // Light & view
        float3 L = normalize(float3(-0.35, 0.45, 0.82));
        float3 V = float3(0.0, 0.0, 1.0);

        float diff = clamp(dot(N, L), 0.0, 1.0);
        float3 H = normalize(L + V);
        float spec = pow(clamp(dot(N, H), 0.0, 1.0), 28.0) * 0.20; // tiny premium sheen

        // Depth shading applied softly over the whole energetic field (not just crest mask)
        float clothMask = clamp(abs(height) * 10.0, 0.0, 1.0);   // where the cloth actually "moves"
        clothMask = mix(clothMask, hero, 0.45);                  // keep visibility of hero

        float ambient = 0.82;
        float shade = ambient + (1.0 - ambient) * diff;

        // Apply: subtle darkening + highlight, controlled
        col *= mix(1.0, shade, 0.18 * clothMask);                // depth (shadowing)
        col = clamp(col + spec * clothMask, 0.0, 1.0);           // sheen

        // Keep temperature shift very subtle (no whitening)
        float3 coolCore = mix(pal.A, float3(0.576, 0.584, 0.776), 0.45);
        float3 warmEdge = mix(pal.B, float3(0.941, 0.871, 0.819), 0.35);
        float3 heroTint = mix(coolCore, warmEdge, sCurve(rN));

        // Slight tint only where cloth moves
        col = screenBlend(col, heroTint, clothMask * 0.10);

        // Keep saturation controlled
        col = desat(col, 0.95);
    }

    // Micro grain
    float grain = (noise21(uv01 * res * 0.65 + t * 0.25) - 0.5) * 0.0070;
    col = clamp(col + grain, 0.0, 1.0);

    return col;
}

// ============================================================
// Fragment (direct crossfade)
// ============================================================

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant AdaptiveBackgroundUniforms& u [[buffer(0)]]) {
    float2 res = max(u.resolution, float2(1.0));
    float2 uv01 = in.position.xy / res;
    float2 uv = uv01 - 0.5;
    uv.x *= (res.x / res.y);

    float t = u.time;

    // smooth, direct crossfade
    float e = sCurve(clamp(u.transition, 0.0, 1.0));

    float3 a = renderOne(uv, uv01, res, t, u.fromMood);
    float3 b = renderOne(uv, uv01, res, t, u.toMood);

    float3 col = mix(a, b, e);
    return float4(col, 1.0);
}
