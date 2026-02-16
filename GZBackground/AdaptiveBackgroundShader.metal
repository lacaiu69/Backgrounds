#include <metal_stdlib>
using namespace metal;

// ============================================================
//  ADAPTIVE BACKGROUND (3 moods) — CLEAN + CONSISTENT
//  moods: 0 = Soothing  -> cloud volumes (VISIBLE, slow, non-linear drift)
//         1 = Neutral   -> glass sweep (subtle, premium)
//         2 = Energetic -> wonky cloth ripples (DARK, 3D feel)
//
//  Notes:
//  - Direct crossfade between fromMood/toMood (no stepping through neutral).
// ============================================================

struct AdaptiveBackgroundUniforms {
    float time;
    float pad0;
    float2 resolution;
    float fromMood;
    float toMood;
    float transition; // 0..1
    float pad1;
};

struct VertexOut { float4 position [[position]]; };

vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    float2 pos[6] = {
        {-1,-1},{ 1,-1},{-1, 1},
        {-1, 1},{ 1,-1},{ 1, 1}
    };
    VertexOut o;
    o.position = float4(pos[vid], 0, 1);
    return o;
}

// ------------------------------------------------------------
// Utils
// ------------------------------------------------------------
static inline float sCurve(float x) { return x*x*(3.0 - 2.0*x); }

static inline float3 desat(float3 c, float keep) {
    float l = dot(c, float3(0.299, 0.587, 0.114));
    return mix(float3(l), c, keep);
}

static inline float2 rot(float2 p, float a) {
    float s = sin(a), c = cos(a);
    return float2(c*p.x - s*p.y, s*p.x + c*p.y);
}

static inline float3 screenBlend(float3 base, float3 top, float a) {
    float3 s = 1.0 - (1.0 - base) * (1.0 - top);
    return mix(base, s, a);
}

// ------------------------------------------------------------
// Noise (value noise + fbm)
// ------------------------------------------------------------
static inline float hash21(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453123);
}
static inline float hash11(float x) {
    return fract(sin(x * 127.1) * 43758.5453123);
}
static inline float noise21(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);

    float a = hash21(i);
    float b = hash21(i + float2(1, 0));
    float c = hash21(i + float2(0, 1));
    float d = hash21(i + float2(1, 1));

    float2 u = f*f*(3.0 - 2.0*f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

static inline float fbm4(float2 p) {
    float f = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) {
        f += a * noise21(p);
        p = p * 2.02 + float2(11.3, 7.7);
        a *= 0.5;
    }
    return f;
}

static inline float fbm6(float2 p) {
    float f = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) {
        f += a * noise21(p);
        p = p * 2.02 + float2(17.1, 9.2);
        a *= 0.5;
    }
    return f;
}

// Approx gradient of noise field (for wrap / fake lighting)
static inline float2 fieldGradNoise(float2 p) {
    const float e = 0.0035;
    float cx = noise21(p + float2(e, 0.0)) - noise21(p - float2(e, 0.0));
    float cy = noise21(p + float2(0.0, e)) - noise21(p - float2(0.0, e));
    return float2(cx, cy) / (2.0*e);
}

// ------------------------------------------------------------
// Palettes
// ------------------------------------------------------------
struct Palette { float3 A; float3 B; float3 C; };

static inline Palette palSoothing() {
    Palette p;
    p.A = float3(0.576, 0.584, 0.776); // #9395C6
    p.B = float3(0.514, 0.584, 0.671); // #8395AB
    p.C = float3(0.769, 0.808, 0.769); // #C4CEC4
    return p;
}

static inline Palette palNeutral() {
    Palette p;
    p.A = float3(0.925, 0.894, 0.882); // #EFE4E1
    p.B = float3(0.882, 0.816, 0.796); // #E1D0CB
    p.C = float3(0.769, 0.808, 0.769); // #C4CEC4
    return p;
}

static inline Palette palEnergetic() {
    Palette p;
    p.A = float3(0.972, 0.915, 0.865); // warm cream
    p.B = float3(0.910, 0.745, 0.720); // blush
    p.C = float3(0.576, 0.584, 0.776); // warm shadow
    return p;
}

// ------------------------------------------------------------
// Animated base gradient (ALL moods)
// ------------------------------------------------------------
// Global light gray anchor
//constant float3 LIGHT_GRAY = float3(0.929, 0.929, 0.929);
// ------------------------------------------------------------


// ------------------------------------------------------------
// Value Noise
// ------------------------------------------------------------
static inline float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);

    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));

    float2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
           (c - a) * u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}
static inline float3 baseGradientAnimated(float2 q,
                                          float2 uv,
                                          float t,
                                          Palette pal,
                                          float moodEnergy)
{
    const float3 LIGHT_GRAY = float3(0.929, 0.929, 0.929); // #EDEDED

    float2 drift1 = float2(0.10*sin(t*0.08 + 1.2), 0.08*cos(t*0.07 + 2.0));
    float2 drift2 = float2(0.06*sin(t*0.05 + 0.4), 0.05*cos(t*0.06 + 0.9));
    float2 qq = q + drift1 + drift2 * (0.6 + 0.4*moodEnergy);

    float g1 = smoothstep(-0.80, 0.80, qq.y + 0.10*sin(t*0.05));
    float g2 = smoothstep(-0.95, 0.95, qq.x * 0.62 + qq.y * 0.20 + 0.06*cos(t*0.04));
    float m  = 0.62 * g1 + 0.38 * g2;

    float3 base = (m < 0.5)
        ? mix(pal.A, pal.B, m * 2.0)
        : mix(pal.B, pal.C, (m - 0.5) * 2.0);

    /// --------------------------------------------------------
    // EXPANDED + NON-LINEAR TOP GRAY
    // --------------------------------------------------------

    float y = -uv.y;   // normalized so top = larger values

    // Expand region
    float grayStart = 0.0;     // starts at mid screen
    float grayEnd   = 0.5;     // full strength at very top

    float mask = smoothstep(grayStart, grayEnd, y);
    float breakup = 0.02 * noise(q * 0.8 + t * 0.02);
    mask = clamp(mask + breakup, 0.0, 1.0);
 

    // Mood-dependent intensity
    float grayStrength = mix(0.90, 0.70, sCurve(moodEnergy));

    float3 finalColor = mix(base, LIGHT_GRAY, mask * grayStrength);

    return finalColor;
}
static inline float2 hash22(float2 p) {
    float n = sin(dot(p, float2(127.1, 311.7)));
    return fract(float2(43758.5453123, 22578.1459123) * n);
}

// Soft circle glow with gaussian-ish falloff
static inline float glow(float2 p, float r) {
    float d = length(p);
    float x = d / max(r, 1e-4);
    return exp(-x * x * 2.2); // wide + blurry
}

static inline float2 heroFireflies(float2 q, float t) {
    // Keep everything slow + non-linear
    float tt = t * 0.22;
    tt += 0.35 * sin(tt * 0.55) + 0.18 * sin(tt * 0.93 + 1.7);

    // Two depth layers: far (bigger, slower, dimmer) + near (smaller, slightly brighter)
    float mask = 0.0;
    float depth = 0.0;

    // Control knobs
    const int COUNT = 15;              // 6..12 feels good
    const float fieldScale = 0.75;    // spatial distribution
    const float drift = 0.22;         // movement amplitude
    const float baseRadFar  = 0.125;   // glow size
    const float baseRadNear = 0.085;

    for (int i = 0; i < COUNT; i++) {
        float fi = float(i) + 1.0;

        // Stable random per particle
        float2 seed = float2(fi, fi * 1.37);
        float2 r0 = hash22(seed);
        float2 r1 = hash22(seed + 19.2);

        // Anchor positions ([-0.5..0.5] space-ish)
        float2 anchor = (r0 - 0.5) * fieldScale;

        // Slow drifting (curvy, not linear)
        float phA = (r1.x * 6.2831);
        float phB = (r1.y * 6.2831);

        float2 wv = float2(
            sin(tt * (0.55 + 0.25 * r1.x) + phA),
            cos(tt * (0.48 + 0.22 * r1.y) + phB)
        );

        // Slightly different motion per “depth”
        float isNear = (hash11(fi * 3.9) > 0.55) ? 1.0 : 0.0;
        float z = mix(0.35, 0.85, isNear); // depth 0..1

        float2 pos = anchor + wv * drift * mix(0.65, 1.05, z);

        // Appear/disappear envelope (soft, non-stroby, desynced)
        float blink = 0.5 + 0.5 * sin(tt * (0.85 + 0.35 * r1.x) + phB + fi);
        blink = smoothstep(0.25, 0.95, blink); // long fades

        // Occasional “rest” so it doesn’t feel constant
        float gate = fbm4(float2(tt * (0.18 + 0.05 * r1.y), fi * 2.7));
        gate = smoothstep(0.35, 0.80, gate);

        float env = blink * mix(0.55, 1.0, gate);

        // Radius + intensity by depth
        float rad = mix(baseRadFar, baseRadNear, z);
        float inten = mix(0.42, 0.95, z) * env;

        // Add contribution
        float g = glow(q - pos, rad) * inten;

        mask += g;
        depth += g * z; // weighted depth
    }

    // Normalize/shape
    mask = clamp(mask, 0.0, 1.0);
    depth = (mask > 1e-4) ? clamp(depth / mask, 0.0, 1.0) : 0.5;

    // Keep it subtle: compress highlights so it never “sparkles”
    mask = pow(mask, 1.35);

    return float2(mask, depth);
}

// ============================================================
// HERO 0 — SOOTHING CLOUD VOLUMES (VISIBLE, slow, non-linear)
// Returns: heroMask (0..1), wrap (0..1)
// ============================================================
static inline float2 heroCloudVolumes(float2 uv, float t) {
    // Slow, non-linear time warp (prevents "linear scroll feeling")
    // MUCH slower base time
    float tn = t * 0.28;

    // Softer non-linear drift (lower amplitude + lower frequency)
    tn += 0.20 * sin(tn * 0.06)
        + 0.10 * sin(tn * 0.12 + 1.7);
    float2 dirA = normalize(float2(0.75, -0.30));
    float2 dirB = normalize(float2(-0.40, -0.60));

    float tt = tn * 0.14;

    // 3 depth layers (parallax)
    float2 pFar  = uv * 0.72 + dirA * (tt * 0.38);
    float2 pMid  = uv * 1.02 + dirA * (tt * 0.62) + float2(0.22, -0.15);
    float2 pNear = uv * 1.42 + dirB * (tt * 0.86) + float2(-0.18, 0.10);

    // Curved drift (non-linear motion)
    float swirl = (fbm4(uv * 0.82 + float2(tt * 0.22, -tt * 0.16)) - 0.5);
    float2 swirlV = float2(-uv.y, uv.x) * (0.13 * swirl);
    pMid  += swirlV;
    pNear += swirlV * 1.55;

    float fFar  = fbm6(pFar  * 0.92);
    float fMid  = fbm6(pMid  * 1.02);
    float fNear = fbm6(pNear * 1.06);

    // Lower thresholds -> more visible shapes
    float vFar  = smoothstep(0.40, 0.83, fFar);
    float vMid  = smoothstep(0.38, 0.82, fMid);
    float vNear = smoothstep(0.36, 0.81, fNear);

    float hero = 0.0;
    hero += vMid  * 0.72;
    hero += vNear * 0.68;
    hero += vFar  * 0.40;

    // Intermittency (subtle gating)
    float env = 0.60 + 0.40 * sin(tn * 0.11 + 1.2);
    env = smoothstep(0.18, 0.95, env);

    float ir = fbm4(float2(tn * 0.040, 7.9));
    ir = smoothstep(0.40, 0.86, ir);

    float envelope = mix(0.62, 1.0, env * 0.65 + ir * 0.35);
    hero *= envelope;

    // Wrap lighting (fake volume)
    float2 g = fieldGradNoise(pMid * 0.52);
    float3 n = normalize(float3(-g.x * 0.24, -g.y * 0.24, 1.0));
    float3 L = normalize(float3(-0.55, 0.65, 0.50));
    float wrap = clamp(dot(n, L) * 0.5 + 0.5, 0.0, 1.0);
    wrap = smoothstep(0.22, 0.95, wrap);

    hero = pow(clamp(hero, 0.0, 1.0), 0.95);

    return float2(hero, wrap);
}

// ============================================================
// HERO 1 — NEUTRAL GLASS SWEEP
// ============================================================
static inline float heroGlassSweep(float2 q, float t) {
    float2 p = rot(q, 0.55);
    float cx = -0.70 + fract(t * 0.06) * 1.40;
    float d  = abs(p.x - cx);

    float core = exp(-(d*d) / (0.09*0.09));
    float tail = exp(-(d*d) / (0.22*0.22)) * 0.55;

    float micro = fbm4(q * 2.2 + float2(0.0, t * 0.07));
    micro = smoothstep(0.55, 0.92, micro) * 0.22;

    return clamp(core + tail + micro, 0.0, 1.0);
}

// ============================================================
// HERO 2 — ENERGETIC WONKY CLOTH RIPPLES (DARK, 3D feel)
// Returns: heroMask, wrap
// ============================================================
static inline float2 heroClothRipplesWonky(float2 q, float t) {
    float tt = t * 0.45;

    float2 dir1 = normalize(float2(0.92, 0.38));
    float2 dir2 = normalize(float2(-0.55, 0.83));
    float2 dir3 = normalize(float2(0.15, -0.99));

    float w = fbm6(q * 1.05 + float2(tt*0.10, -tt*0.08));
    w = (w - 0.5) * 1.15;

    float s1 = dot(q, dir1);
    float s2 = dot(q, dir2);
    float s3 = dot(q, dir3);

    float f1 = sin(s1 * 8.5  - tt * 1.25 + w * 1.2);
    float f2 = sin(s2 * 10.2 - tt * 1.05 + w * 1.0);
    float f3 = sin(s3 * 6.8  - tt * 0.90 + w * 0.9);

    float rip = (f1*0.55 + f2*0.45 + f3*0.35);
    rip = 0.5 + 0.5 * rip;

    float ridges  = smoothstep(0.58, 0.92, rip);
    float valleys = smoothstep(0.50, 0.86, 1.0 - rip);
    float hero = mix(ridges, valleys, 0.42);

    float packet = 0.5 + 0.5 * sin((s1 + s2)*0.85 - tt*0.25 + w*0.6);
    packet = smoothstep(0.30, 0.90, packet);
    hero *= mix(0.55, 1.0, packet);

    float patch = fbm4(q * 1.15 + float2(tt*0.06, -tt*0.05));
    patch = smoothstep(0.30, 0.85, patch);
    hero *= mix(0.70, 1.0, patch);

    hero = pow(clamp(hero, 0.0, 1.0), 1.10);

    float2 g = fieldGradNoise(q * 0.65 + float2(tt*0.05, -tt*0.04));
    float3 n = normalize(float3(-g.x * 0.22, -g.y * 0.22, 1.0));
    float3 L = normalize(float3(-0.35, 0.55, 0.75));
    float wrap = clamp(dot(n, L) * 0.5 + 0.5, 0.0, 1.0);
    wrap = smoothstep(0.20, 0.98, wrap);

    return float2(hero, wrap);
}

// ============================================================
// Render one mood
// ============================================================
static inline float3 renderOne(float2 uv, float2 uv01, float2 res, float t, float mood) {

    Palette pal = (mood < 0.5) ? palSoothing()
               : (mood < 1.5) ? palNeutral()
                              : palEnergetic();

    float moodEnergy = (mood < 0.5) ? 0.20 : ((mood < 1.5) ? 0.35 : 0.60);

    float tt = t * mix(0.030, 0.12, sCurve(moodEnergy));

    float2 warp = float2(
        noise21(uv * 1.05 + float2(0.0,  tt * 0.35)),
        noise21(uv * 1.05 + float2(2.1, -tt * 0.30))
    );
    warp = (warp - 0.5) * mix(0.018, 0.040, sCurve(moodEnergy));
    float2 q = uv + warp;

    float3 col = baseGradientAnimated(q, uv, t, pal, moodEnergy);

    // soft internal bloom
    {
        float2 c = float2(0.16 * sin(tt * 0.55), 0.14 * cos(tt * 0.48));
        float d = length(q - c);
        float bloom = exp(-(d*d) / (0.42*0.42));
        col = mix(col, mix(pal.B, pal.A, 0.40), bloom * 0.055);
    }

    if (mood < 0.5) {
        // ✅ SOOTHING = CLOUD VOLUMES (the visible one)
        float2 hw = heroCloudVolumes(q * 1.12, t);
        float hero = hw.x;
        float wrap = hw.y;

        float3 cloudLight = mix(pal.B, pal.A, 0.52);
        float3 cloudDepth = mix(pal.C, pal.B, 0.50);

        float a = hero * (0.55 + 0.55 * wrap);
        col = screenBlend(col, cloudLight, a);

        // depth (prevents flat “milky” look)
        col = mix(col, cloudDepth, hero * (0.07 + 0.05 * (1.0 - wrap)));
        col = clamp(col + hero * 0.008, 0.0, 1.0);

        // keep color so clouds don't vanish
        col = desat(col, 0.955);

    } else if (mood < 1.5) {
        // NEUTRAL = glass sweep + fireflies (soft, sparse, premium)
        float glass = heroGlassSweep(q, tt);
        glass = pow(glass, 0.92);

        float2 ff = heroFireflies(q * 1.02, t);
        float fire = ff.x;
        float z    = ff.y;

        // Tint: keep inside neutral palette; slightly cooler for far, slightly warmer for near
        float3 farTint  = mix(pal.B, pal.C, 0.35);
        float3 nearTint = mix(pal.A, pal.B, 0.55);
        float3 fireTint = mix(farTint, nearTint, sCurve(z));

        // Fireflies are "behind glass": use screen blend but low alpha
        col = screenBlend(col, fireTint, fire * 0.65);
        col = clamp(col + fire * 0.010, 0.0, 1.0);

        // Keep your glass sweep as the “hero motion” baseline
        float3 glassTint = mix(pal.B, pal.A, 0.55);
        col = screenBlend(col, glassTint, glass * 0.11);
        col = clamp(col + glass * 0.012, 0.0, 1.0);

        col = desat(col, 0.945);
    }
else {
        // ENERGETIC (kept as in your working version)
        float2 hw = heroClothRipplesWonky(q * 1.05, t);
        float hero = pow(hw.x, 0.85);
        float wrap = hw.y;

        float3 blushLight = mix(pal.B, pal.A, 0.55);

        // DARKEN along ripples (depth cue)
        col *= (1.0 - hero * 0.32);

        // wrap lighting (no whitening)
        float wrapLift = hero * (0.07 + 0.06 * wrap);
        col = screenBlend(col, blushLight, wrapLift);

        col = clamp(col + hero * 0.010, 0.0, 1.0);
        col = desat(col, 0.96);
    }

    // Micro grain
    float grain = (noise21(uv01 * res * 0.55 + t * 0.22) - 0.5) * 0.0060;
    col = clamp(col + grain, 0.0, 1.0);

    return col;
}

// ============================================================
// Fragment — direct crossfade
// ============================================================
fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant AdaptiveBackgroundUniforms& u [[buffer(0)]])
{
    float2 res = max(u.resolution, float2(1.0));
    float2 uv01 = in.position.xy / res;
    float2 uv = uv01 - 0.5;
    uv.x *= (res.x / res.y);

    float t = u.time;
    float e = sCurve(clamp(u.transition, 0.0, 1.0));

    float3 a = renderOne(uv, uv01, res, t, u.fromMood);
    float3 b = renderOne(uv, uv01, res, t, u.toMood);

    return float4(mix(a, b, e), 1.0);
}
