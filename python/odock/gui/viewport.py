# SPDX-License-Identifier: GPL-3.0-or-later
"""The ModernGL viewport of the OpenDocking workbench.

Rendering strategy
------------------
Atoms are drawn as *impostor spheres*: one camera-facing quad per atom, with the
sphere baked into the fragment shader and ``gl_FragDepth`` written from the
analytic sphere surface. That gives correct sphere/sphere interpenetration and
sphere-quality shading for the price of two triangles per atom — which is what
makes a 20 000-atom receptor interactive from Python (ModernGL issues the same
GL calls a C extension would).

Three more passes complete the picture:

* a **triangle mesh** (position, normal, colour) built once on the CPU when the
  style changes, for bond cylinders, the secondary-structure cartoon, the flat
  ribbon and the plain tube (:func:`build_group_mesh`);
* a **coloured-line pass** for interaction dashes, measurements and the axes;
* two **overlay** sphere passes drawn straight from the scene every frame — the
  selection and the translucent reference pose — plus a camera-space front
  cut-away (``Scene.front_clip``) that opens a deep pocket so a ligand bound
  inside it is not hidden by the protein around it.

The search box is a wireframe cube drawn with the same instancing machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - only importable with the gui extra
    import moderngl
except Exception:  # pragma: no cover
    moderngl = None  # type: ignore[assignment]

from .structure import Atom, element_color, element_radius

SPHERE_VS = """
#version 330
in vec2 in_quad;
in vec3 in_center;
in vec4 in_radius_color;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_view_center;
out vec3 v_color;
out float v_radius;
out vec2 v_quad;

void main() {
    v_color = in_radius_color.yzw;
    v_radius = in_radius_color.x;
    v_quad = in_quad;

    vec4 view_center = u_view * vec4(in_center, 1.0);
    v_view_center = view_center.xyz;

    // Billboard in clip space. Because the quad corner offset must shrink with
    // depth, it is multiplied by w -- which is exactly the perspective divide
    // that would otherwise undo it. u_proj[1][1] is 1/tan(fov/2), so the quad
    // ends up exactly `1.35 * radius` Angstrom across on screen.
    vec4 clip = u_proj * view_center;
    clip.xy += in_quad * v_radius * 1.35 * u_proj[1][1];
    gl_Position = clip;
}
"""

SPHERE_FS = """
#version 330
in vec3 v_view_center;
in vec3 v_color;
in float v_radius;
in vec2 v_quad;

uniform mat4 u_proj;
uniform vec3 u_light_view;
uniform float u_alpha;

// Camera-space front clip: fragments nearer to the camera than this are cut
// away. It is how a ligand buried in a space-filling pocket stays visible —
// see Scene.front_clip. Zero disables it.
uniform float u_clip_front;

// How far the geometry is pushed towards grey and towards the background:
// 0 = untouched, 1 = fully dimmed. Used to recede the parts of the structure
// that are not part of the interaction site being emphasised.
uniform float u_dim;

out vec4 fragColor;

/// Desaturate and darken a colour by `amount`.
vec3 dimmed(vec3 color, float amount) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(color, vec3(luma), amount * 0.85) * (1.0 - amount * 0.45);
}

void main() {
    float r2 = dot(v_quad, v_quad);
    if (r2 > 1.0) discard;
    // Front clip, per *atom*: an atom whose centre is in front of the plane is
    // dropped whole, so the pocket wall opens a clean gap onto the ligand
    // instead of showing the hollow inside of a half-cut sphere.
    if (u_clip_front > 0.0 && -v_view_center.z < u_clip_front) discard;
    float z = sqrt(1.0 - r2);
    vec3 normal = vec3(v_quad, z);

    // Analytic sphere depth: the near surface is `radius * z` closer to the
    // camera, so spheres interpenetrate correctly instead of z-fighting.
    vec4 point = u_proj * vec4(v_view_center + normal * v_radius, 1.0);
    gl_FragDepth = clamp((point.z / point.w) * 0.5 + 0.5, 0.0, 1.0);

    float diff = max(dot(normal, normalize(u_light_view)), 0.0);
    float spec = pow(diff, 24.0) * 0.35;
    float rim = pow(1.0 - z, 2.5) * 0.30;
    vec3 color = dimmed(v_color, u_dim) * (0.28 + 0.78 * diff) + spec + rim;
    fragColor = vec4(color, u_alpha);
}
"""

BOX_VS = """
#version 330
in vec3 in_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""

#: Coloured primitives (interaction dashes, measurement lines, axes).
LINE_VS = """
#version 330
in vec3 in_pos;
in vec4 in_color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

LINE_FS = """
#version 330
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }
"""

#: Real triangles: bond cylinders, the protein cartoon and the flat ribbon.
#:
#: Unlike the sphere pass this is ordinary geometry — the ribbons and sticks are
#: swept surfaces built once on the CPU (see :func:`build_mesh`) and drawn with
#: one ``glDrawArrays(GL_TRIANGLES)``. The normal is rotated into view space with
#: ``u_view`` so the shading matches the impostor spheres exactly, and the
#: absolute value of the diffuse term lights back faces too, because a ribbon has
#: no meaningful inside.
MESH_VS = """
#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;

uniform mat4 u_mvp;
uniform mat4 u_view;

out vec3 v_normal_view;
out vec3 v_color;
out vec3 v_view_pos;

void main() {
    v_color = in_color;
    v_normal_view = mat3(u_view) * in_normal;
    v_view_pos = (u_view * vec4(in_pos, 1.0)).xyz;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

MESH_FS = """
#version 330
in vec3 v_normal_view;
in vec3 v_color;
in vec3 v_view_pos;

uniform vec3 u_light_view;
uniform float u_alpha;

// Camera-space front clip, shared with the sphere pass (0 disables it).
uniform float u_clip_front;
// Desaturation applied to the parts of the structure that are not the focus
// (see SPHERE_FS).
uniform float u_dim;

out vec4 fragColor;

vec3 dimmed(vec3 color, float amount) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(color, vec3(luma), amount * 0.85) * (1.0 - amount * 0.45);
}

void main() {
    if (u_clip_front > 0.0 && -v_view_pos.z < u_clip_front) discard;
    vec3 normal = normalize(v_normal_view);
    vec3 light = normalize(u_light_view);
    float diff = abs(dot(normal, light));
    float spec = pow(diff, 24.0) * 0.28;
    float rim = pow(1.0 - diff, 3.0) * 0.12;
    fragColor = vec4(dimmed(v_color, u_dim) * (0.34 + 0.72 * diff) + spec + rim, u_alpha);
}
"""

#: Screen-space ambient occlusion, applied to the scene colour buffer.
#:
#: The depth buffer holds non-linear window depth, where a fixed epsilon means
#: wildly different world distances across the scene, so the shader linearises
#: it first and then compares in Ångström. The occlusion radius grows with
#: distance so that far-away geometry is not over-darkened.
SSAO_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_color;
uniform sampler2D u_depth;
uniform vec2 u_texel;
uniform float u_near;
uniform float u_far;
uniform float u_strength;
out vec4 fragColor;

float linear_depth(vec2 uv) {
    float d = texture(u_depth, uv).r;
    if (d >= 0.99999) return u_far;            // background
    float z = d * 2.0 - 1.0;
    return (2.0 * u_near * u_far) / (u_far + u_near - z * (u_far - u_near));
}

void main() {
    vec3 base = texture(u_color, v_uv).rgb;
    float z = linear_depth(v_uv);
    if (z >= u_far * 0.99) {
        fragColor = vec4(base, 1.0);
        return;
    }

    const vec2 kernel[12] = vec2[12](
        vec2( 0.20236, 0.54132), vec2(-0.44721, 0.31379),
        vec2( 0.61803, -0.23607), vec2(-0.15643, -0.58779),
        vec2( 0.95106,  0.30902), vec2(-0.80902,  0.58779),
        vec2( 0.30902, -0.95106), vec2(-0.58779, -0.80902),
        vec2( 0.00000,  1.00000), vec2( 0.70711,  0.70711),
        vec2(-1.00000,  0.00000), vec2( 0.00000, -1.00000)
    );

    // The sample radius is ~1.2 A at the focus distance and scales with depth.
    float radius = u_texel.x * 0.012 * z;
    float bias = 0.06 * z;

    float occlusion = 0.0;
    for (int i = 0; i < 12; ++i) {
        float zn = linear_depth(v_uv + kernel[i] * radius);
        float delta = z - zn;                  // >0: the neighbour is in front
        occlusion += (delta > bias && delta < z * 0.35) ? 1.0 : 0.0;
    }
    occlusion /= 12.0;
    float ao = clamp(1.0 - occlusion * u_strength * 0.85, 0.35, 1.0);
    fragColor = vec4(base * ao, 1.0);
}
"""

FULLSCREEN_VS = """
#version 330
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv * 0.5 + 0.5;
    gl_Position = vec4(in_uv, 0.0, 1.0);
}
"""

#: Colour per interaction type, as required by the project brief module E.2.
INTERACTION_COLORS = {
    "hbond": (0.20, 0.90, 0.95, 0.95),        # cyan
    "salt_bridge": (0.95, 0.25, 0.85, 0.95),  # magenta
    "pi_pi": (0.25, 0.90, 0.35, 0.95),        # green
    "cation_pi": (0.98, 0.60, 0.15, 0.95),    # orange
    "hydrophobic": (0.62, 0.64, 0.68, 0.75),  # grey
    "clash": (1.00, 0.15, 0.15, 1.00),        # red
}

#: Human-readable label per interaction type (used by the legend and the table).
INTERACTION_LABELS = {
    "hbond": "hydrogen bond",
    "salt_bridge": "salt bridge",
    "pi_pi": "pi-pi stacking",
    "cation_pi": "cation-pi",
    "hydrophobic": "hydrophobic",
    "clash": "steric clash",
}

BOX_FS = """
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""


# ---------------------------------------------------------------------------
# representation styles
# ---------------------------------------------------------------------------

#: Every protein representation the renderer understands.
#:
#: ``cartoon``   secondary structure: helix ribbons, strand arrows, loop tubes
#: ``ribbon``    one smooth flat ribbon along the C-alpha trace, rainbow-tinted
#: ``tube``      the plain thick C-alpha polyline
#: ``spheres``   van der Waals spheres scaled down to a ball-and-stick look
#: ``spacefill`` real van der Waals radii at 1:1 (CPK), for reading a pocket
#: ``sticks``    thin cylinders along the perceived bonds
#: ``ball_stick``spheres on the atoms plus bond cylinders
#: ``dots``      a sparse sphere sampling, for a very large structure
PROTEIN_STYLES: Tuple[str, ...] = (
    "cartoon",
    "ribbon",
    "tube",
    "spheres",
    "spacefill",
    "sticks",
    "ball_stick",
    "dots",
)

#: Every small-molecule representation the renderer understands.
#:
#: ``ball_stick`` spheres on the atoms plus two-tone bond cylinders
#: ``sticks``     licorice: bond cylinders with round joints, no atom spheres
#: ``wireframe``  the skeletal formula: thin bonds, heteroatom dots only
#: ``spheres``    van der Waals spheres scaled down for a ball-and-stick view
#: ``spacefill``  real van der Waals radii at 1:1 (CPK): the ligand is drawn
#:                exactly as large as it is, so the pocket atoms whose surfaces
#:                interpenetrate it are directly readable
LIGAND_STYLES: Tuple[str, ...] = (
    "ball_stick",
    "sticks",
    "wireframe",
    "spheres",
    "spacefill",
)

#: Legacy names kept working. ``style_receptor``/``style_ligand`` are the names
#: the first version of the workbench used and several tests still set them, so
#: assigning one of these maps onto the canonical style above.
_PROTEIN_ALIASES = {
    "cartoon": "cartoon",
    "ribbon": "ribbon",
    "tube": "tube",
    "spheres": "spheres",
    "spacefill": "spacefill",
    "space_filling": "spacefill",
    "space-filling": "spacefill",
    "cpk": "spacefill",
    "vdw": "spacefill",
    "sticks": "sticks",
    "ball_stick": "ball_stick",
    "dots": "dots",
    "ball": "ball_stick",       # legacy ligand wording
    "stick": "sticks",
    "wireframe": "wireframe",
}
_LIGAND_ALIASES = {
    "ball": "ball_stick",
    "ball_stick": "ball_stick",
    "stick": "sticks",
    "sticks": "sticks",
    "wireframe": "wireframe",
    "spheres": "spheres",
    "spacefill": "spacefill",
    "space_filling": "spacefill",
    "space-filling": "spacefill",
    "cpk": "spacefill",
    "vdw": "spacefill",
    "cartoon": "ball_stick",    # a ligand has no cartoon; keep rendering
}


def canon_protein_style(value: str) -> str:
    """Map any accepted protein-style spelling onto a canonical one."""
    return _PROTEIN_ALIASES.get(str(value).strip().lower(), "spheres")


def canon_ligand_style(value: str) -> str:
    """Map any accepted ligand-style spelling onto a canonical one."""
    return _LIGAND_ALIASES.get(str(value).strip().lower(), "ball_stick")


#: The six faces of the search box as ``(corner indices counter-clockwise seen
#: from outside, outward normal)``. Only the faces turned towards the camera are
#: filled, so the translucent box stays one layer thick whatever the viewpoint.
_BOX_FACES = (
    ((0, 3, 2, 1), (0.0, 0.0, -1.0)),
    ((4, 5, 6, 7), (0.0, 0.0, 1.0)),
    ((0, 1, 5, 4), (0.0, -1.0, 0.0)),
    ((3, 7, 6, 2), (0.0, 1.0, 0.0)),
    ((0, 4, 7, 3), (-1.0, 0.0, 0.0)),
    ((1, 2, 6, 5), (1.0, 0.0, 0.0)),
)

#: How much brighter and larger a focused (interacting) atom is drawn than the
#: ball-and-stick ball it sits on. Generous on purpose: the emphasis has to read
#: on top of a dimmed protein, and the base spheres of the focused atoms are
#: suppressed so nothing competes with it.
FOCUS_BOOST = 1.38
FOCUS_PADDING = 0.06
#: The two atoms a contact actually names are drawn brighter still, so the pair
#: behind each dash is identifiable at a glance.
FOCUS_KEY_BOOST = 1.65


def focus_color(element: str) -> Tuple[float, float, float]:
    """A saturated, brighter version of an element's colour.

    The interaction emphasis has to survive being looked at next to a dimmed
    protein, so the CPK hue is pushed towards full saturation and lifted.
    """
    red, green, blue = element_color(element)
    return (
        min(1.0, red * 1.45 + 0.20),
        min(1.0, green * 1.45 + 0.20),
        min(1.0, blue * 1.45 + 0.20),
    )


def focus_key_color(element: str) -> Tuple[float, float, float]:
    """The colour of the two atoms a contact names: hot, but still hued."""
    red, green, blue = element_color(element)
    return (
        min(1.0, 0.52 + red * 0.72),
        min(1.0, 0.52 + green * 0.72),
        min(1.0, 0.52 + blue * 0.72),
    )


@dataclass
class Camera:
    """An orbit camera."""

    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 40.0
    azimuth: float = 0.6
    elevation: float = 0.35
    fov: float = 45.0

    def eye(self) -> Tuple[float, float, float]:
        ce = math.cos(self.elevation)
        return (
            self.target[0] + self.distance * ce * math.cos(self.azimuth),
            self.target[1] + self.distance * ce * math.sin(self.azimuth),
            self.target[2] + self.distance * math.sin(self.elevation),
        )

    def view(self) -> np.ndarray:
        eye = np.array(self.eye(), dtype="f4")
        tgt = np.array(self.target, dtype="f4")
        up = np.array([0.0, 0.0, 1.0], dtype="f4")
        f = tgt - eye
        n = np.linalg.norm(f)
        f = f / (n if n > 1e-9 else 1.0)
        s = np.cross(f, up)
        n = np.linalg.norm(s)
        if n < 1e-6:
            s = np.array([1.0, 0.0, 0.0], dtype="f4")
        else:
            s = s / n
        u = np.cross(s, f)
        m = np.eye(4, dtype="f4")
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -float(np.dot(s, eye))
        m[1, 3] = -float(np.dot(u, eye))
        m[2, 3] = float(np.dot(f, eye))
        return m

    def projection(self, width: int, height: int) -> np.ndarray:
        aspect = max(width, 1) / max(height, 1)
        near = max(0.05, self.distance * 0.005)
        far = self.distance * 20.0 + 500.0
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        m = np.zeros((4, 4), dtype="f4")
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2.0 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    def frame(self, lo: Sequence[float], hi: Sequence[float], margin: float = 1.6) -> None:
        """Point the camera at a bounding box and back off far enough to see it."""
        center = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
        extent = max(hi[i] - lo[i] for i in range(3))
        self.target = tuple(center)  # type: ignore[assignment]
        self.distance = max(8.0, extent * margin)


@dataclass
class Scene:
    """The renderable content."""

    receptor: List[Atom] = None  # type: ignore[assignment]
    ligand: List[Atom] = None  # type: ignore[assignment]
    show_receptor: bool = True
    show_ligand: bool = True
    #: Ligand spheres are deliberately large: the pose is the point of the view.
    ball_scale: float = 0.55
    receptor_scale: float = 0.20
    box: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    receptor_cutoff: Optional[Tuple[float, float, float]] = None
    receptor_radius: float = 0.0

    # -- appearance ---------------------------------------------------------
    #: One of :data:`PROTEIN_STYLES`.
    style_protein: str = "spheres"
    #: One of :data:`LIGAND_STYLES`.
    style_ligand: str = "ball_stick"
    show_axes: bool = False
    ssao: bool = True
    ssao_strength: float = 1.1

    # -- annotations --------------------------------------------------------
    #: :class:`odock.analysis.Interaction` objects, drawn as dashed lines.
    interactions: List[object] = None  # type: ignore[assignment]
    #: Ligand atom indices to highlight (e.g. the picked atom).
    highlight: List[int] = None  # type: ignore[assignment]
    #: (point_a, point_b, label) measurement segments in Å.
    measurements: List[tuple] = None  # type: ignore[assignment]
    #: Rotatable bonds the user has locked: [(i, j), ...]; drawn highlighted.
    locked_bonds: List[tuple] = None  # type: ignore[assignment]
    #: Perceived bonds of the ligand as :class:`odock.gui.bonds.Bond` objects.
    #: They unpack as ``(a, b)`` like the old tuple pairs did.
    ligand_bonds: List[object] = None  # type: ignore[assignment]
    #: Perceived bonds of the receptor, same shape.
    receptor_bonds: List[object] = None  # type: ignore[assignment]

    # -- selection and the pose reference ----------------------------------
    #: Atoms the user has selected, as ``[("receptor" | "ligand", index), ...]``.
    #: They are re-drawn brighter with a radius boost so a selection is visible
    #: even when the atom is buried in a space-filling surface. The default is
    #: empty, and an empty selection draws nothing at all.
    selection: List[tuple] = None  # type: ignore[assignment]
    #: A reference ligand drawn semi-transparently behind the current pose —
    #: the first pose (or the native ligand) while another pose is displayed,
    #: so the movement between the two is visible. Empty means "no overlay".
    ghost_ligand: List[Atom] = None  # type: ignore[assignment]
    #: Alpha of the reference overlay. Low enough to see through, high enough
    #: to read its shape.
    ghost_alpha: float = 0.35
    #: Sphere scale of the overlay (1.0 = the same radii as the ligand style).
    ghost_scale: float = 1.0
    #: Cut away everything between the camera and the ligand. A ligand bound in
    #: a deep pocket is *inside* the protein, so a space-filling receptor hides
    #: it completely; this camera-space front clip (it follows the camera, so it
    #: stays correct while orbiting) removes exactly the material in front of
    #: the ligand and leaves the pocket wall around it visible.
    #:
    #: It is **off by default and must stay a deliberate choice**: the plane is
    #: placed relative to the *current* ligand, so while it is on two poses of
    #: the same protein are cut away differently. ``frame_binding_site`` no
    #: longer enables it automatically for exactly that reason.
    front_clip: bool = False

    # -- the search box -----------------------------------------------------
    #: Opacity of the translucent box fill, 0.0–1.0. ``0.0`` hides the fill and
    #: keeps the edges, so the search volume is never invisible. The default is
    #: deliberately light: the fill is depth-tested, but at a close camera the
    #: box covers most of the viewport and anything heavier washes the structure
    #: out. The edges are drawn at ``box_alpha + 0.45`` so they stay readable
    #: under a faint fill.
    box_alpha: float = 0.22
    #: The six face-centre drag grips. Off by default: the box is edited from
    #: the Grid workbench, not by dragging its faces, so the orange spheres were
    #: only ever visual noise. Kept behind a flag for a future grips feature.
    show_box_handles: bool = False

    # -- interaction emphasis ----------------------------------------------
    #: The interactions to emphasise, normally ``analysis.Interaction`` objects
    #: (the same ones ``interactions`` holds). Every residue the receptor side
    #: (``.a``) belongs to is redrawn as ball-and-stick, the ligand atoms on the
    #: other side (``.b``) with them, both in a saturated highlight colour, and
    #: the rest of the structure is dimmed — so "how does this pose bind?"
    #: reads at a glance instead of hiding in a crowd of spheres. Empty means
    #: "no emphasis", and the scene is drawn exactly as before.
    interaction_focus: List[object] = None  # type: ignore[assignment]
    #: How far the non-focused part of the structure recedes, 0.0–1.0.
    focus_dim: float = 0.72

    def __setattr__(self, name: str, value) -> None:
        # ``style_receptor`` is the historical name of ``style_protein`` and
        # ``ball``/``stick`` are the historical ligand styles: accepting them
        # here keeps every existing caller working while the canonical names
        # above stay the ones the renderer branches on.
        if name == "style_receptor":
            name = "style_protein"
        if name == "style_protein":
            value = canon_protein_style(value)
        elif name == "style_ligand":
            value = canon_ligand_style(value)
        object.__setattr__(self, name, value)

    @property
    def style_receptor(self) -> str:
        """Alias of :attr:`style_protein` (the pre-cartoon name)."""
        return self.style_protein

    def __post_init__(self) -> None:
        self.receptor = self.receptor or []
        self.ligand = self.ligand or []
        self.interactions = self.interactions or []
        self.highlight = self.highlight or []
        self.measurements = self.measurements or []
        self.locked_bonds = self.locked_bonds or []
        self.ligand_bonds = self.ligand_bonds or []
        self.receptor_bonds = self.receptor_bonds or []
        self.selection = self.selection or []
        self.ghost_ligand = self.ghost_ligand or []
        self.interaction_focus = self.interaction_focus or []

    def has_content(self) -> bool:
        return bool(self.receptor or self.ligand)

    def bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        pts = [a.position for a in self.receptor] + [a.position for a in self.ligand]
        if not pts:
            return ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        arr = np.asarray(pts, dtype="f8")
        return tuple(arr.min(axis=0)), tuple(arr.max(axis=0))

    def visible_receptor(self) -> List[Atom]:
        """Receptor atoms to draw, honouring the optional distance cut-off.

        Drawing an entire 30 000-atom protein is both slow and unreadable; the
        cut-off keeps only what surrounds the ligand (or the box).
        """
        if self.receptor_cutoff is None or self.receptor_radius <= 0:
            return list(self.receptor)
        cx, cy, cz = self.receptor_cutoff
        r2 = self.receptor_radius * self.receptor_radius
        out = []
        for a in self.receptor:
            dx = a.x - cx
            dy = a.y - cy
            dz = a.z - cz
            if dx * dx + dy * dy + dz * dz <= r2:
                out.append(a)
        return out


# ---------------------------------------------------------------------------
# mesh geometry, built on the CPU when the style changes
# ---------------------------------------------------------------------------
#
# Everything below is plain NumPy: a style switch rebuilds one interleaved
# (position, normal, colour) vertex array, which is then a single draw call.
# Nothing here touches OpenGL, so the geometry can be tested — and inspected —
# without a context.


#: Secondary-structure colours, in the spirit of the usual viewers: helices red,
#: strands yellow, loops blue-grey. The brightness also ramps along the chain so
#: the fold direction is readable in a still image.
SS_HELIX = "H"
SS_STRAND = "E"
SS_COIL = "C"
_SS_COLORS = {
    SS_HELIX: (0.88, 0.33, 0.34),
    SS_STRAND: (0.96, 0.80, 0.28),
    SS_COIL: (0.52, 0.70, 0.88),
}

#: Cross-section half-width / half-thickness, in Å, per secondary structure.
_SS_SHAPE = {
    SS_HELIX: (1.05, 0.30),
    SS_STRAND: (1.15, 0.24),
    SS_COIL: (0.38, 0.38),
}

#: The number of triangles around a swept cross-section. Ten reads as round at
#: any zoom the workbench offers; the sticks use fewer because a receptor can
#: have thousands of them.
RIBBON_SIDES = 10
STICK_SIDES = 8
RECEPTOR_STICK_SIDES = 6

#: Ball-and-stick ball size: the factor ``_pack`` applies to a van der Waals
#: radius. ``SPHERE_VS`` draws the impostor 1.35 x the radius it is handed (the
#: billboard must cover the ray-cast sphere), so 0.13 puts a carbon ball at
#: about 0.30 Å on screen — big enough to read as an atom, small enough that the
#: bond cylinders stay visible between the two coloured halves of a bond.
BALL_STICK_SCALE = 0.13


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def dihedral(p0, p1, p2, p3) -> float:
    """The IUPAC dihedral angle p0-p1-p2-p3 in degrees."""
    b0 = np.asarray(p0, dtype="f8") - np.asarray(p1, dtype="f8")
    b1 = np.asarray(p2, dtype="f8") - np.asarray(p1, dtype="f8")
    b2 = np.asarray(p3, dtype="f8") - np.asarray(p2, dtype="f8")
    norm = float(np.linalg.norm(b1))
    if norm < 1e-8:
        return 0.0
    axis = b1 / norm
    v = b0 - float(np.dot(b0, axis)) * axis
    w = b2 - float(np.dot(b2, axis)) * axis
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(axis, v), w))
    return math.degrees(math.atan2(y, x))


def ca_trace(atoms):
    """The C-alpha trace of a protein, or the P trace of a nucleic acid.

    Returns ``(points (m, 3), residues, kept)`` where ``residues`` holds the
    ``N``/``CA``/``C``/``O`` coordinates of every residue (for the backbone
    dihedrals) and ``kept`` lists which residue each trace point came from.

    The element is checked as well as the name: a calcium ion is called ``CA``
    in a PDBQT file and must never be mistaken for a C-alpha.
    """
    order: dict = {}
    residues: List[dict] = []
    for atom in atoms:
        key = (
            str(getattr(atom, "chain", "") or ""),
            int(getattr(atom, "res_id", 0) or 0),
            str(getattr(atom, "res_name", "") or ""),
        )
        slot = order.get(key)
        if slot is None:
            slot = len(residues)
            order[key] = slot
            residues.append({})
        name = str(getattr(atom, "name", "") or "").strip().upper()
        element = str(getattr(atom, "element", "") or "")
        if name in ("N", "CA", "C", "O") and element in ("C", "N", "O"):
            residues[slot].setdefault(
                name, (float(atom.x), float(atom.y), float(atom.z))
            )
        elif name == "P" and element == "P":
            residues[slot].setdefault(name, (float(atom.x), float(atom.y), float(atom.z)))

    points: List[tuple] = []
    kept: List[int] = []
    for slot, residue in enumerate(residues):
        if "CA" in residue:
            points.append(residue["CA"])
            kept.append(slot)
        elif "P" in residue:
            points.append(residue["P"])
            kept.append(slot)
    if not points:
        return np.zeros((0, 3), dtype="f8"), residues, kept
    return np.asarray(points, dtype="f8"), residues, kept


def backbone_torsions(residues, kept) -> List[Tuple[Optional[float], Optional[float]]]:
    """φ/ψ for every trace point, from the backbone N/CA/C atoms.

    A residue with an incomplete backbone (a gap in the chain, a terminal
    residue) yields ``None`` for the missing dihedral, which the secondary
    structure assignment treats as "unknown" rather than as a contradiction.
    """
    out: List[Tuple[Optional[float], Optional[float]]] = []
    for index, slot in enumerate(kept):
        residue = residues[slot] if 0 <= slot < len(residues) else {}
        phi = psi = None
        previous = (
            residues[kept[index - 1]]
            if index > 0 and 0 <= kept[index - 1] < len(residues)
            else None
        )
        following = (
            residues[kept[index + 1]]
            if index + 1 < len(kept) and 0 <= kept[index + 1] < len(residues)
            else None
        )
        try:
            if previous and "C" in previous and {"N", "CA", "C"} <= set(residue):
                phi = dihedral(previous["C"], residue["N"], residue["CA"], residue["C"])
            if following and "N" in following and {"N", "CA", "C"} <= set(residue):
                psi = dihedral(residue["N"], residue["CA"], residue["C"], following["N"])
        except Exception:  # pragma: no cover - defensive
            phi = psi = None
        out.append((phi, psi))
    return out


#: DSSP-lite tolerances. The C-alpha distance pattern is the primary signal
#: because it survives a missing backbone atom; the dihedrals only confirm it.
#:
#: * α-helix: d(i, i+3) ∈ [4.4, 5.8] Å and d(i, i+4) ∈ [5.5, 7.2] Å
#:   (ideal 5.0 / 6.2 Å), φ ∈ [-140, -20]°, ψ ∈ [-80, 25]° (ideal -57 / -47).
#: * β-strand: d(i, i+2) ∈ [6.0, 7.4] Å and d(i, i+3) ≥ 8.6 Å
#:   (ideal 6.7 / 10.0 Å), φ ∈ [-170, -70]°, ψ ≥ 80° (ideal -120 / 130).
#:
#: Runs shorter than ``min_helix``/``min_strand`` residues are demoted to coil,
#: so a single distorted residue cannot grow a fake helix.
HELIX_MIN_RUN = 4
STRAND_MIN_RUN = 2


def _in_range(value, low, high) -> bool:
    return value is not None and low <= value <= high


def secondary_structure(points, torsions=None) -> str:
    """Assign ``H`` (helix), ``E`` (strand) or ``C`` (coil) to every residue.

    Documented tolerances are in the module source; the C-alpha distance
    pattern decides, and the backbone dihedrals can only confirm it (a residue
    whose dihedrals are unknown is not penalised).
    """
    count = len(points)
    if count == 0:
        return ""
    raw = [SS_COIL] * count
    for i in range(count):
        helix = strand = False
        if i + 4 < count:
            d3 = float(np.linalg.norm(points[i + 3] - points[i]))
            d4 = float(np.linalg.norm(points[i + 4] - points[i]))
            helix = 4.4 <= d3 <= 5.8 and 5.5 <= d4 <= 7.2
        if i + 2 < count:
            d2 = float(np.linalg.norm(points[i + 2] - points[i]))
            d3b = (
                float(np.linalg.norm(points[i + 3] - points[i]))
                if i + 3 < count
                else 10.0
            )
            strand = 6.0 <= d2 <= 7.4 and d3b >= 8.6
        if torsions is not None and i < len(torsions):
            phi, psi = torsions[i]
            # A missing dihedral (a gap in the backbone) means "unknown", not
            # "contradiction": only a dihedral we actually have may overrule the
            # C-alpha distance pattern, which is the primary signal.
            if helix and phi is not None and psi is not None:
                if not (
                    _in_range(phi, -140.0, -20.0) and _in_range(psi, -80.0, 25.0)
                ):
                    helix = _in_range(phi, -160.0, -20.0)
            if strand and phi is not None and psi is not None:
                if not (
                    _in_range(phi, -170.0, -70.0)
                    and (psi >= 80.0 or psi <= -140.0)
                ):
                    strand = _in_range(phi, -170.0, -70.0) and _in_range(
                        psi, 80.0, 180.0
                    )
        raw[i] = SS_HELIX if helix else (SS_STRAND if strand else SS_COIL)

    # Demote short runs: a lone turn is a coil, not a helix.
    raw = _filter_runs(raw, SS_HELIX, HELIX_MIN_RUN)
    raw = _filter_runs(raw, SS_STRAND, STRAND_MIN_RUN)
    # A real helix extends one residue past its last strict C-alpha contact.
    return _extend(raw, SS_HELIX)


def _filter_runs(codes: List[str], kind: str, minimum: int) -> List[str]:
    out = list(codes)
    start = None
    for index in range(len(out) + 1):
        inside = index < len(out) and out[index] == kind
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            if index - start < minimum:
                for slot in range(start, index):
                    out[slot] = SS_COIL
            start = None
    return out


def _extend(codes: List[str], kind: str) -> List[str]:
    out = list(codes)
    for index in range(len(out)):
        if out[index] != kind:
            continue
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(out) and out[neighbour] == SS_COIL:
                # Only grow into a coil that is not itself the start of a
                # different assignment.
                out[neighbour] = kind
    return out


def catmull_rom(points, samples_per_segment: int = 6):
    """A smooth curve through ``points`` (uniform Catmull-Rom).

    Returns ``(positions, source)`` where ``source[i]`` is the index of the
    input point the sample came from — that is how the secondary-structure code
    and the colours follow the smoothed trace.
    """
    pts = np.asarray(points, dtype="f8").reshape(-1, 3)
    count = len(pts)
    if count == 0:
        return pts, np.zeros(0, dtype=int)
    if count == 1:
        return pts.copy(), np.zeros(1, dtype=int)
    steps = max(2, int(samples_per_segment))
    t = np.linspace(0.0, 1.0, steps, endpoint=False).reshape(-1, 1)
    chunks = []
    source = []
    for i in range(count - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, count - 1)]
        curve = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
        )
        chunks.append(curve)
        source.extend([i] * steps)
    chunks.append(pts[-1:].copy())
    source.append(count - 1)
    return np.concatenate(chunks, axis=0), np.asarray(source, dtype=int)


def rotation_minimising_frames(points: np.ndarray):
    """``(tangent, u, v)`` along a polyline, transported without spinning.

    A parallel-transported frame is what keeps a cartoon ribbon from flipping
    over between residues: along a helix ``u`` stays roughly radial (so the
    ribbon wraps around the coil), and along a β-strand it stays near the sheet
    normal (so the arrow reads as a flat blade).
    """
    pts = np.asarray(points, dtype="f8").reshape(-1, 3)
    count = len(pts)
    if count == 1:
        tangent = np.array([[0.0, 0.0, 1.0]])
        u = np.array([[1.0, 0.0, 0.0]])
        return tangent, u, np.cross(tangent, u)

    tangent = np.empty_like(pts)
    tangent[0] = pts[1] - pts[0]
    tangent[-1] = pts[-1] - pts[-2]
    if count > 2:
        tangent[1:-1] = pts[2:] - pts[:-2]
    tangent = _unit(tangent)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(tangent[0][2])) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = reference - tangent[0] * float(np.dot(reference, tangent[0]))
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:  # pragma: no cover - degenerate
        first = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    u = np.empty_like(pts)
    u[0] = first / norm
    for i in range(1, count):
        axis = tangent[i]
        projected = u[i - 1] - axis * float(np.dot(u[i - 1], axis))
        length = float(np.linalg.norm(projected))
        if length < 1e-8:  # pragma: no cover - degenerate
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(float(axis[0])) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            projected = fallback - axis * float(np.dot(fallback, axis))
            length = float(np.linalg.norm(projected)) or 1.0
        u[i] = projected / length
    return tangent, u, np.cross(tangent, u)


def _sweep(
    points: np.ndarray,
    half_width: np.ndarray,
    half_thickness: np.ndarray,
    colors: np.ndarray,
    sides: int = RIBBON_SIDES,
    frames=None,
) -> np.ndarray:
    """Sweep an elliptical cross-section along a curve, as triangles.

    The cross-section is an ellipse of the given half width / half thickness in
    the transported frame, so a helix becomes a ribbon wrapped round the coil
    and a loop becomes a thin tube — one continuous surface for the whole chain,
    with the shape blended across secondary-structure boundaries.
    """
    pts = np.asarray(points, dtype="f8").reshape(-1, 3)
    count = len(pts)
    if count < 2:
        return np.zeros((0, 9), dtype="f4")
    widths = np.maximum(np.asarray(half_width, dtype="f8").reshape(-1), 1e-3)
    thickness = np.maximum(np.asarray(half_thickness, dtype="f8").reshape(-1), 1e-3)
    colors = np.asarray(colors, dtype="f8").reshape(-1, 3)

    _, u, v = frames if frames is not None else rotation_minimising_frames(pts)
    theta = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos = np.cos(theta)[None, :, None]
    sin = np.sin(theta)[None, :, None]

    ring = (
        pts[:, None, :]
        + (widths[:, None, None] * cos) * u[:, None, :]
        + (thickness[:, None, None] * sin) * v[:, None, :]
    )
    normals = (cos / widths[:, None, None]) * u[:, None, :] + (
        sin / thickness[:, None, None]
    ) * v[:, None, :]
    normals = _unit(normals)

    vertices = np.concatenate(
        [
            ring,
            normals,
            np.broadcast_to(colors[:, None, :], ring.shape),
        ],
        axis=2,
    ).reshape(-1, 9)

    i_index = np.arange(count - 1)[:, None]
    j_index = np.arange(sides)[None, :]
    j_next = (j_index + 1) % sides
    base = i_index * sides
    a = (base + j_index).ravel()
    b = (base + j_next).ravel()
    c = (base + sides + j_next).ravel()
    d = (base + sides + j_index).ravel()
    triangles = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
    )
    return np.ascontiguousarray(vertices[triangles].reshape(-1, 9), dtype="f4")


def rainbow(fraction: float) -> Tuple[float, float, float]:
    """A blue → cyan → green → yellow → red ramp for the plain ribbon style."""
    fraction = min(1.0, max(0.0, float(fraction)))
    hue = 0.66 * (1.0 - fraction)
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    q = 1.0 - f
    table = [
        (1.0, f, 0.0), (q, 1.0, 0.0), (0.0, 1.0, f),
        (0.0, q, 1.0), (f, 0.0, 1.0), (1.0, 0.0, q),
    ]
    return table[i % 6]


def _blend(values: np.ndarray, passes: int = 2) -> np.ndarray:
    """Smooth a per-sample parameter along the trace (1-2-1 kernel)."""
    out = np.asarray(values, dtype="f8")
    for _ in range(passes):
        padded = np.concatenate([out[:1], out, out[-1:]])
        out = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return out


def _group_sizes(codes, scale: float = 1.0):
    """Per-sample half width / half thickness for a secondary-structure string."""
    widths = np.array([_SS_SHAPE[code][0] for code in codes], dtype="f8") * scale
    thickness = np.array([_SS_SHAPE[code][1] for code in codes], dtype="f8") * scale
    return widths, thickness


def _color_ramp(codes, count: int, index: np.ndarray) -> np.ndarray:
    """Secondary-structure colours, brightened along the chain."""
    ramp = 0.74 + 0.36 * (index / max(1, count - 1))
    colors = np.empty((len(codes), 3), dtype="f8")
    for i, code in enumerate(codes):
        base = np.array(_SS_COLORS.get(code, _SS_COLORS[SS_COIL]), dtype="f8")
        colors[i] = np.clip(base * ramp[i], 0.0, 1.0)
    return colors


def _arrow_heads(points, codes, frames, colors, half_width) -> np.ndarray:
    """Flat arrow blades at the end of every β-strand run.

    A separate piece of geometry (not spliced into the main sweep) so the
    ribbon can continue into the following loop without a stretched triangle.
    """
    tangent, u, v = frames
    pieces = []
    index = 0
    count = len(codes)
    while index < count:
        if codes[index] != SS_STRAND:
            index += 1
            continue
        end = index
        while end + 1 < count and codes[end + 1] == SS_STRAND:
            end += 1
        if end > index:  # a one-residue strand has no room for a head
            base = points[end]
            direction = tangent[end]
            blade = base + np.array([0.05, 0.55, 1.15])[:, None] * direction[None, :]
            blade_width = np.array([max(2.0, half_width[end] * 1.7), 1.45, 0.10])
            blade_thickness = np.array([0.24, 0.22, 0.10])
            blade_colors = np.repeat(colors[end][None, :], 3, axis=0)
            frames_local = (
                np.repeat(direction[None, :], 3, axis=0),
                np.repeat(u[end][None, :], 3, axis=0),
                np.repeat(v[end][None, :], 3, axis=0),
            )
            pieces.append(
                _sweep(
                    blade,
                    blade_width,
                    blade_thickness,
                    blade_colors,
                    sides=6,
                    frames=frames_local,
                )
            )
        index = end + 1
    if not pieces:
        return np.zeros((0, 9), dtype="f4")
    return np.concatenate(pieces, axis=0)


def build_cartoon_mesh(atoms) -> np.ndarray:
    """The secondary-structure cartoon: helix ribbons, strand arrows, loops."""
    points, residues, kept = ca_trace(atoms)
    if len(points) < 2:
        return np.zeros((0, 9), dtype="f4")
    torsions = backbone_torsions(residues, kept)
    codes = secondary_structure(points, torsions)

    smooth, source = catmull_rom(points, samples_per_segment=6)
    smooth = _smooth_positions(smooth, source, passes=1)
    codes_per_sample = [codes[i] if i < len(codes) else SS_COIL for i in source]
    widths, thickness = _group_sizes(codes_per_sample)
    widths = _blend(widths, 2)
    thickness = _blend(thickness, 2)
    colors = _color_ramp(
        codes_per_sample, len(points), np.asarray(source, dtype="f8")
    )
    frames = rotation_minimising_frames(smooth)
    body = _sweep(smooth, widths, thickness, colors, sides=RIBBON_SIDES, frames=frames)
    heads = _arrow_heads(smooth, codes_per_sample, frames, colors, widths)
    if len(heads):
        return np.ascontiguousarray(np.concatenate([body, heads], axis=0), dtype="f4")
    return body


def _smooth_positions(points: np.ndarray, source: np.ndarray, passes: int = 1) -> np.ndarray:
    """A gentle 1-2-1 smoothing of the sampled trace (not of the CA positions).

    The spline already passes through every C-alpha; this only removes the
    small kinks a uniform Catmull-Rom leaves at the knots, which matters for a
    flat ribbon because a kink shows as a crease.
    """
    out = np.asarray(points, dtype="f8").copy()
    for _ in range(max(0, passes)):
        padded = np.concatenate([out[:1], out, out[-1:]], axis=0)
        out = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return out


def build_ribbon_mesh(atoms) -> np.ndarray:
    """One smooth flat ribbon along the C-alpha trace, rainbow-tinted."""
    points, _residues, _kept = ca_trace(atoms)
    if len(points) < 2:
        return np.zeros((0, 9), dtype="f4")
    smooth, source = catmull_rom(points, samples_per_segment=6)
    smooth = _smooth_positions(smooth, source, passes=1)
    count = len(smooth)
    widths = np.full(count, 1.15)
    thickness = np.full(count, 0.26)
    colors = np.array([rainbow(i / max(1, count - 1)) for i in range(count)])
    return _sweep(smooth, widths, thickness, colors, sides=RIBBON_SIDES)


def build_tube_mesh(atoms, radius: float = 1.05) -> np.ndarray:
    """A round tube swept along the smoothed C-alpha trace.

    The plain "thick backbone" look: no secondary-structure shaping at all, just
    one continuous round surface. An elliptical cross-section with equal
    half-axes *is* a circle, so :func:`_sweep` already builds the tube.
    """
    points, _residues, _kept = ca_trace(atoms)
    if len(points) < 2:
        return np.zeros((0, 9), dtype="f4")
    smooth, source = catmull_rom(points, samples_per_segment=6)
    smooth = _smooth_positions(smooth, source, passes=1)
    count = len(smooth)
    radii = np.full(count, float(radius))
    colors = np.tile(np.asarray(element_color("C"), dtype="f8"), (count, 1))
    return _sweep(smooth, radii, radii, colors, sides=RIBBON_SIDES)


def _cylinders(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    sides: int = STICK_SIDES,
    caps: bool = True,
) -> np.ndarray:
    """Tube geometry for a batch of segments (bond halves)."""
    starts = np.asarray(starts, dtype="f8").reshape(-1, 3)
    ends = np.asarray(ends, dtype="f8").reshape(-1, 3)
    if len(starts) == 0:
        return np.zeros((0, 9), dtype="f4")
    radii = np.maximum(np.asarray(radii, dtype="f8").reshape(-1), 1e-3)
    colors = np.asarray(colors, dtype="f8").reshape(-1, 3)

    delta = ends - starts
    axis = _unit(delta)
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(axis), 1))
    near_z = np.abs(axis[:, 2]) > 0.9
    reference[near_z] = np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(axis, reference))
    v = np.cross(axis, u)

    theta = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos = np.cos(theta)[None, :, None]
    sin = np.sin(theta)[None, :, None]
    radial = cos * u[:, None, :] + sin * v[:, None, :]
    normals = _unit(radial)
    ring = radii[:, None, None] * radial

    first = starts[:, None, :] + ring
    second = ends[:, None, :] + ring
    color_columns = np.broadcast_to(colors[:, None, :], first.shape)

    vertex_blocks = [
        np.concatenate([first, normals, color_columns], axis=2),
        np.concatenate([second, normals, color_columns], axis=2),
    ]
    if caps:
        cap_normals = np.concatenate([-axis[:, None, :], axis[:, None, :]], axis=1)
        cap_colors = np.concatenate([colors[:, None, :], colors[:, None, :]], axis=1)
        cap_points = np.concatenate(
            [starts[:, None, :], ends[:, None, :]], axis=1
        )
        vertex_blocks.append(np.concatenate([cap_points, cap_normals, cap_colors], axis=2))

    vertices = np.concatenate(vertex_blocks, axis=1).reshape(-1, 9)
    side_stride = sides * 2
    per_cylinder = side_stride + (2 if caps else 0)

    j = np.arange(sides)
    j_next = (j + 1) % sides
    k = np.arange(len(starts))
    base = (k * per_cylinder)[:, None]
    a = (base + j).ravel()
    b = (base + j_next).ravel()
    c = (base + sides + j_next).ravel()
    d = (base + sides + j).ravel()
    triangles = [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    if caps:
        first_center = base[:, 0] + side_stride
        last_center = base[:, 0] + side_stride + 1
        triangles.append(
            np.stack([a, b, np.repeat(first_center, sides)], axis=1)
        )
        triangles.append(
            np.stack([c, d, np.repeat(last_center, sides)], axis=1)
        )
    indices = np.concatenate(triangles, axis=0)
    return np.ascontiguousarray(vertices[indices].reshape(-1, 9), dtype="f4")


def _bond_endpoints(bonds):
    """``(a, b, order)`` triples from Bond objects or plain ``(a, b)`` pairs."""
    out = []
    for bond in bonds or ():
        if hasattr(bond, "a") and hasattr(bond, "b"):
            out.append((int(bond.a), int(bond.b), int(getattr(bond, "order", 1) or 1)))
        else:
            try:
                first, second = bond[0], bond[1]
            except Exception:  # pragma: no cover - defensive
                continue
            out.append((int(first), int(second), 1))
    return out


def build_bond_mesh(
    atoms,
    bonds,
    *,
    radius: float = 0.13,
    sides: int = STICK_SIDES,
    caps: bool = False,
    double_offset: float = 0.09,
    hide_carbon_hydrogen: bool = False,
    index_map=None,
) -> np.ndarray:
    """Two-tone cylinders for every perceived bond.

    A bond is split at its midpoint: each half carries the CPK colour of the
    atom it starts from, which is the standard ball-and-stick look and makes it
    obvious *which* atoms a stick joins. A double bond becomes two thinner,
    parallel cylinders offset in a plane perpendicular to the bond.
    """
    positions = np.array(
        [(float(a.x), float(a.y), float(a.z)) for a in atoms], dtype="f8"
    ).reshape(-1, 3)
    if len(positions) == 0:
        return np.zeros((0, 9), dtype="f4")
    symbols = [str(getattr(a, "element", "") or "C") for a in atoms]
    palette = np.array([element_color(s) for s in symbols], dtype="f8")

    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    radii: List[float] = []
    for i, j, order in _bond_endpoints(bonds):
        if index_map is not None:
            if not (0 <= i < len(index_map) and 0 <= j < len(index_map)):
                continue
            i = int(index_map[i])
            j = int(index_map[j])
            if i < 0 or j < 0:
                continue
        if not (0 <= i < len(positions) and 0 <= j < len(positions)):
            continue
        if hide_carbon_hydrogen:
            pair = {symbols[i], symbols[j]}
            if pair == {"H", "C"}:
                continue
        middle = 0.5 * (positions[i] + positions[j])
        strands = max(1, min(int(order or 1), 3))
        if strands == 2:
            bond_axis = _unit((positions[j] - positions[i]).reshape(1, 3))[0]
            reference = np.array([0.0, 0.0, 1.0])
            if abs(float(bond_axis[2])) > 0.9:
                reference = np.array([1.0, 0.0, 0.0])
            offset = _unit(np.cross(bond_axis, reference).reshape(1, 3))[0] * double_offset
            shifts = (offset, -offset)
            half_radius = radius * 0.62
        else:
            shifts = (np.zeros(3),)
            half_radius = radius
        for shift in shifts:
            starts.append(positions[i] + shift)
            ends.append(middle + shift)
            colors.append(palette[i])
            radii.append(half_radius)
            starts.append(positions[j] + shift)
            ends.append(middle + shift)
            colors.append(palette[j])
            radii.append(half_radius)

    if not starts:
        return np.zeros((0, 9), dtype="f4")
    return _cylinders(
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(radii),
        np.asarray(colors),
        sides=sides,
        caps=caps,
    )


def build_group_mesh(
    atoms,
    bonds,
    style: str,
    *,
    index_map=None,
    sides: int = STICK_SIDES,
) -> np.ndarray:
    """The triangle mesh for one style of one group (``[]`` when not needed).

    ``index_map`` maps a bond's atom index onto a position in ``atoms`` so the
    receptor distance cut-off can drop bonds that leave the visible set.
    """
    if style == "cartoon":
        return build_cartoon_mesh(atoms)
    if style == "ribbon":
        return build_ribbon_mesh(atoms)
    if style == "tube":
        return build_tube_mesh(atoms)
    if style == "ball_stick":
        return build_bond_mesh(
            atoms, bonds, radius=0.12, sides=sides, caps=False, index_map=index_map
        )
    if style == "sticks":
        return build_bond_mesh(
            atoms, bonds, radius=0.15, sides=sides, caps=False, index_map=index_map
        )
    if style == "wireframe":
        return build_bond_mesh(
            atoms,
            bonds,
            radius=0.075,
            sides=sides,
            caps=True,
            double_offset=0.07,
            hide_carbon_hydrogen=True,
            index_map=index_map,
        )
    return np.zeros((0, 9), dtype="f4")


class Renderer:
    """A small instanced-sphere renderer."""
    def __init__(self, ctx, scene: Scene) -> None:
        self.ctx = ctx
        self.scene = scene
        self.sphere_prog = ctx.program(vertex_shader=SPHERE_VS, fragment_shader=SPHERE_FS)
        self.box_prog = ctx.program(vertex_shader=BOX_VS, fragment_shader=BOX_FS)

        quad = np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], dtype="f4"
        )
        self.quad_buffer = ctx.buffer(quad.tobytes())

        self._receptor_capacity = 0
        self._ligand_capacity = 0
        self.receptor_buffer = None
        self.ligand_buffer = None
        self.receptor_vao = None
        self.ligand_vao = None
        self.receptor_count = 0
        self.ligand_count = 0

        self.box_corners = np.array(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            dtype="f4",
        )
        self.box_buffer = ctx.buffer(self.box_corners.tobytes(), dynamic=True)
        edges = np.array(
            [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7],
            dtype="i4",
        )
        self.box_index_buffer = ctx.buffer(edges.tobytes())
        self.box_vao = ctx.vertex_array(
            self.box_prog,
            [(self.box_buffer, "3f", "in_pos")],
            index_buffer=self.box_index_buffer,
            index_element_size=4,
        )

        self.dirty_receptor = True
        self.dirty_ligand = True
        self._receptor_cached: Optional[List[Atom]] = None

        # -- triangle mesh (bond cylinders, cartoon, ribbon) -----------------
        self.mesh_prog = ctx.program(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
        self.mesh_receptor_buffer = None
        self.mesh_receptor_vao = None
        self.mesh_receptor_vertices = 0
        self.mesh_ligand_buffer = None
        self.mesh_ligand_vao = None
        self.mesh_ligand_vertices = 0
        self._mesh_receptor_capacity = 0
        self._mesh_ligand_capacity = 0

        # -- interaction emphasis (the focused residues and ligand atoms) ----
        # A second, small mesh pair: the focused part of the structure is drawn
        # as ball-and-stick whatever the global style is, on top of a dimmed
        # copy of everything else.
        self.focus_mesh_receptor_buffer = None
        self.focus_mesh_receptor_vao = None
        self.focus_mesh_receptor_vertices = 0
        self.focus_mesh_ligand_buffer = None
        self.focus_mesh_ligand_vao = None
        self.focus_mesh_ligand_vertices = 0
        self._focus_mesh_receptor_capacity = 0
        self._focus_mesh_ligand_capacity = 0
        self._focus_receptor: List[int] = []
        self._focus_ligand: List[int] = []
        #: The atoms an interaction actually names (as opposed to the whole
        #: residue the focus expands to): drawn hottest, so the pair behind each
        #: dash is identifiable.
        self._focus_key_receptor: List[int] = []
        self._focus_key_ligand: List[int] = []
        self._focus_signature = None
        #: How many atoms the current focus covers, for the tests and the logs.
        self.focus_receptor_atoms = 0
        self.focus_ligand_atoms = 0
        self.focus_residues = 0

        # -- contact dashes, as fat tubes rather than 1-px lines ------------
        # A hairline is invisible against a protein, so the interactions are
        # swept as cylinders whose radius is chosen per segment to land on a
        # fixed on-screen width (see Renderer.INTERACTION_PIXELS).
        self.interaction_buffer = None
        self.interaction_vao = None
        self._interaction_capacity = 0
        self.interaction_vertices = 0

        # -- overlays: the selection and the reference (ghost) pose ----------
        # These are drawn from the scene every frame (the selection can change
        # without a style switch), so they get their own growable sphere buffer
        # rather than sharing the 64-instance handle buffer.
        self.overlay_prog = ctx.program(vertex_shader=SPHERE_VS, fragment_shader=SPHERE_FS)
        self._overlay_buffers: Dict[str, object] = {}
        self._overlay_vaos: Dict[str, object] = {}
        self._overlay_capacity: Dict[str, int] = {}
        self.selection_count = 0
        self.ghost_count = 0
        #: The camera-space front clip depth used by the current frame; the
        #: selection overlay reuses it so a cut-away atom cannot be drawn back.
        self._clip_front = 0.0

        # -- coloured lines (interactions, measurements, axes) --------------
        self.line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self._line_capacity = 0
        self.line_buffer = None
        self.line_vao = None

        # -- draggable box handles: small spheres, re-uploaded every frame --
        self.handle_buffer = ctx.buffer(reserve=64 * 28, dynamic=True)
        self.handle_prog = ctx.program(vertex_shader=SPHERE_VS, fragment_shader=SPHERE_FS)
        self.handle_vao = ctx.vertex_array(
            self.handle_prog,
            [
                (self.quad_buffer, "2f", "in_quad"),
                (self.handle_buffer, "3f 4f/i", "in_center", "in_radius_color"),
            ],
        )
        self.handle_positions: List[tuple] = []
        self._handle_uploaded = -1
        #: How many handle spheres the last frame drew (0 with the grips off).
        self.handle_count = 0
        #: How many emphasis spheres the last frame drew.
        self.focus_sphere_count = 0
        #: How many triangles the translucent box fill drew last frame.
        self.box_fill_triangles = 0
        #: The box edge vertex count of the last frame.
        self.box_edge_vertices = 0

        # -- translucent box fill --------------------------------------------
        # Only the three faces turned towards the camera are uploaded each frame
        # (six would blend six layers of the same alpha and look opaque), and
        # the fill is drawn with the depth test off, which is how a translucent
        # volume can sit in front of the atoms without hiding them and without
        # disturbing the depth buffer the contact dashes are tested against.
        self.box_fill_buffer = ctx.buffer(reserve=18 * 3 * 4, dynamic=True)
        self.box_fill_vao = ctx.vertex_array(
            self.box_prog, [(self.box_fill_buffer, "3f", "in_pos")]
        )

        # -- screen-space ambient occlusion ---------------------------------
        self.fs_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=SSAO_FS)
        self.fs_quad = ctx.buffer(
            np.array([[-1, -1], [3, -1], [-1, 3]], dtype="f4").tobytes()
        )
        self.fs_vao = ctx.vertex_array(self.fs_prog, [(self.fs_quad, "2f", "in_uv")])
        self._ssao_fbo = None
        self._ssao_size = (0, 0)
        self._last_view = np.eye(4)
        self._last_proj = np.eye(4)

    # -- uploads -----------------------------------------------------------

    def _ensure_capacity(self, which: str, needed: int) -> None:
        capacity = self._receptor_capacity if which == "receptor" else self._ligand_capacity
        if needed <= capacity and (
            self.receptor_vao if which == "receptor" else self.ligand_vao
        ) is not None:
            return
        capacity = int(2 ** max(10, math.ceil(math.log2(max(needed, 1)))))
        buf = self.ctx.buffer(reserve=28 * capacity, dynamic=True)
        vao = self.ctx.vertex_array(
            self.sphere_prog,
            [
                (self.quad_buffer, "2f", "in_quad"),
                # `/i` marks the attribute as per-instance (divisor 1).
                (buf, "3f 4f/i", "in_center", "in_radius_color"),
            ],
        )
        if which == "receptor":
            if self.receptor_vao is not None:
                self.receptor_vao.release()
            if self.receptor_buffer is not None:
                self.receptor_buffer.release()
            self.receptor_buffer = buf
            self.receptor_vao = vao
            self._receptor_capacity = capacity
        else:
            if self.ligand_vao is not None:
                self.ligand_vao.release()
            if self.ligand_buffer is not None:
                self.ligand_buffer.release()
            self.ligand_buffer = buf
            self.ligand_vao = vao
            self._ligand_capacity = capacity

    #: Floats per mesh vertex: position, normal, colour.
    MESH_STRIDE = 9

    def _upload_mesh(self, which: str, data: np.ndarray) -> None:
        """Upload one rebuilt mesh, growing the buffer only when it must."""
        vertices = 0 if data is None or data.size == 0 else int(data.shape[0])
        if which == "receptor":
            self.mesh_receptor_vertices = vertices
        else:
            self.mesh_ligand_vertices = vertices
        if vertices == 0:
            return
        flat = np.ascontiguousarray(data.reshape(-1), dtype="f4")
        capacity = (
            self._mesh_receptor_capacity
            if which == "receptor"
            else self._mesh_ligand_capacity
        )
        buffer = (
            self.mesh_receptor_buffer if which == "receptor" else self.mesh_ligand_buffer
        )
        if buffer is None or flat.size > capacity:
            if buffer is not None:
                buffer.release()
            capacity = max(flat.size, 9 * 1024)
            buffer = self.ctx.buffer(reserve=int(capacity * 4), dynamic=True)
            vao = self.ctx.vertex_array(
                self.mesh_prog,
                [(buffer, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
            )
            if which == "receptor":
                if self.mesh_receptor_vao is not None:
                    self.mesh_receptor_vao.release()
                self.mesh_receptor_buffer = buffer
                self.mesh_receptor_vao = vao
                self._mesh_receptor_capacity = capacity
            else:
                if self.mesh_ligand_vao is not None:
                    self.mesh_ligand_vao.release()
                self.mesh_ligand_buffer = buffer
                self.mesh_ligand_vao = vao
                self._mesh_ligand_capacity = capacity
        assert buffer is not None
        buffer.write(flat.tobytes())

    def set_interaction_focus(self, residues=None, ligand_atoms=None, *, dim=None) -> dict:
        """Emphasise an interaction site and return what it covers.

        Parameters
        ----------
        residues:
            The receptor side of the contact: atom indices, or
            ``(chain, res_id, res_name)`` residue keys. An atom index pulls in
            **every atom of its residue**, because a side chain is what a
            chemist wants to see as ball-and-stick.
        ligand_atoms:
            Ligand atom indices to emphasise.
        dim:
            Override :attr:`Scene.focus_dim` for this focus.

        The focused part is drawn as ball-and-stick whatever the global style
        is, the rest of the structure is dimmed, and the geometry is rebuilt on
        the next upload — call :meth:`ViewportWidget.refresh` with
        ``upload_receptor=True``. Passing no arguments clears the focus.

        Most callers do not need this: assigning ``scene.interaction_focus`` to
        the ``analysis.Interaction`` objects of the current pose does the same
        thing from ``upload`` (see :meth:`_sync_focus_from_scene`).
        """
        receptor = list(self.scene.receptor)
        by_key: Dict[tuple, List[int]] = {}
        for index, atom in enumerate(receptor):
            by_key.setdefault(
                (
                    str(getattr(atom, "chain", "") or ""),
                    int(getattr(atom, "res_id", 0) or 0),
                    str(getattr(atom, "res_name", "") or ""),
                ),
                [],
            ).append(index)

        seeds: List[int] = []
        for item in residues or ():
            if isinstance(item, tuple) and len(item) >= 2:
                # A (chain, res_id[, res_name]) residue key; be liberal about the
                # length and about a missing name.
                key = (str(item[0]), int(item[1]))
                seeds.extend(
                    index
                    for index, atom in enumerate(receptor)
                    if (
                        str(getattr(atom, "chain", "") or ""),
                        int(getattr(atom, "res_id", 0) or 0),
                    )
                    == key
                )
            else:
                try:
                    seeds.append(int(item))
                except (TypeError, ValueError):
                    continue
        chosen = self._expand_residues(seeds)

        ligand: List[int] = []
        for item in ligand_atoms or ():
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(self.scene.ligand) and index not in ligand:
                ligand.append(index)

        if dim is not None:
            self.scene.focus_dim = float(dim)
        self._focus_receptor = chosen
        self._focus_ligand = ligand
        self._focus_key_receptor = sorted({int(i) for i in seeds if 0 <= int(i) < len(receptor)})
        self._focus_key_ligand = list(ligand)
        self.focus_receptor_atoms = len(chosen)
        self.focus_ligand_atoms = len(ligand)
        self.focus_residues = len(
            {
                (
                    str(getattr(receptor[i], "chain", "") or ""),
                    int(getattr(receptor[i], "res_id", 0) or 0),
                )
                for i in chosen
            }
        )
        self._focus_signature = self._signature()
        self.dirty_receptor = True
        self.dirty_ligand = True
        return {
            "residues": self.focus_residues,
            "receptor_atoms": self.focus_receptor_atoms,
            "ligand_atoms": self.focus_ligand_atoms,
        }

    def clear_interaction_focus(self) -> None:
        """Drop the emphasis and redraw the scene as it was."""
        self.set_interaction_focus((), ())

    def _focus_from_interactions(self):
        """``(receptor atom indices, ligand atom indices)`` from the scene.

        ``scene.interaction_focus`` holds the ``analysis.Interaction`` objects
        of the pose: ``.a`` is a receptor atom index and ``.b`` a ligand one (a
        representative atom for ring and charge contacts, which is why the whole
        residue is pulled in).
        """
        receptor: List[int] = []
        ligand: List[int] = []
        for item in self.scene.interaction_focus or ():
            a = getattr(item, "a", None)
            b = getattr(item, "b", None)
            if a is None or b is None:
                # Also accept a plain (receptor_index, ligand_index) pair.
                try:
                    a, b = int(item[0]), int(item[1])
                except Exception:
                    continue
            try:
                receptor.append(int(a))
                ligand.append(int(b))
            except (TypeError, ValueError):
                continue
        return receptor, ligand

    def _suppressed_atoms(self, which: str) -> set:
        """``id()``s of the atoms the emphasis already draws itself.

        The base pass leaves these out: a focused residue drawn twice (once as a
        receptor sphere, once as ball-and-stick) is exactly what buried the
        emphasis behind the protein's own spheres. The empty set when there is
        no focus, so the normal scene is untouched.
        """
        if which == "receptor":
            indices, atoms = self._focus_receptor, self.scene.receptor
        else:
            indices, atoms = self._focus_ligand, self.scene.ligand
        return {id(atoms[i]) for i in indices if 0 <= i < len(atoms)}

    def _signature(self):
        """A cheap identity for the current focus, so uploads can be skipped."""
        return (tuple(self._focus_receptor), tuple(self._focus_ligand))

    def _expand_residues(self, indices) -> List[int]:
        """Grow a list of receptor atom indices to every atom of their residues.

        An interaction names one representative atom of a side chain (and, for a
        ring or a charged group, one atom of the group), but what a chemist
        wants to see emphasised is the whole residue — so the focus always
        covers the complete residue, side chain included.
        """
        receptor = self.scene.receptor
        by_key: Dict[tuple, List[int]] = {}
        for index, atom in enumerate(receptor):
            by_key.setdefault(
                (
                    str(getattr(atom, "chain", "") or ""),
                    int(getattr(atom, "res_id", 0) or 0),
                    str(getattr(atom, "res_name", "") or ""),
                ),
                [],
            ).append(index)
        out: List[int] = []
        seen = set()
        for index in indices:
            if not (0 <= index < len(receptor)):
                continue
            key = (
                str(getattr(receptor[index], "chain", "") or ""),
                int(getattr(receptor[index], "res_id", 0) or 0),
                str(getattr(receptor[index], "res_name", "") or ""),
            )
            for atom_index in by_key.get(key, (index,)):
                if atom_index not in seen:
                    seen.add(atom_index)
                    out.append(atom_index)
        return out

    def _sync_focus_from_scene(self) -> bool:
        """Recompute the focus from the scene; True when it changed."""
        if not self.scene.interaction_focus:
            receptor: List[int] = []
            ligand: List[int] = []
            keys_receptor: List[int] = []
            keys_ligand: List[int] = []
        else:
            raw_receptor, raw_ligand = self._focus_from_interactions()
            receptor = self._expand_residues(raw_receptor)
            ligand = sorted({i for i in raw_ligand if 0 <= i < len(self.scene.ligand)})
            keys_receptor = sorted({i for i in raw_receptor if 0 <= i < len(self.scene.receptor)})
            keys_ligand = list(ligand)
        self._focus_receptor = receptor
        self._focus_ligand = ligand
        self._focus_key_receptor = keys_receptor
        self._focus_key_ligand = keys_ligand
        self.focus_receptor_atoms = len(receptor)
        self.focus_ligand_atoms = len(ligand)
        self.focus_residues = len(
            {
                (
                    str(getattr(self.scene.receptor[i], "chain", "") or ""),
                    int(getattr(self.scene.receptor[i], "res_id", 0) or 0),
                )
                for i in receptor
            }
        )
        wanted = self._signature()
        if wanted == self._focus_signature:
            return False
        self._focus_signature = wanted
        return True

    def _upload_focus_mesh(self, which: str, data: np.ndarray) -> None:
        """Upload one focus mesh (the same 9-float vertex format)."""
        vertices = 0 if data is None or data.size == 0 else int(data.shape[0])
        if which == "receptor":
            self.focus_mesh_receptor_vertices = vertices
        else:
            self.focus_mesh_ligand_vertices = vertices
        if vertices == 0:
            return
        flat = np.ascontiguousarray(data.reshape(-1), dtype="f4")
        capacity = (
            self._focus_mesh_receptor_capacity
            if which == "receptor"
            else self._focus_mesh_ligand_capacity
        )
        buffer = (
            self.focus_mesh_receptor_buffer
            if which == "receptor"
            else self.focus_mesh_ligand_buffer
        )
        if buffer is None or flat.size > capacity:
            if buffer is not None:
                buffer.release()
            capacity = max(flat.size, 9 * 512)
            buffer = self.ctx.buffer(reserve=int(capacity * 4), dynamic=True)
            vao = self.ctx.vertex_array(
                self.mesh_prog,
                [(buffer, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
            )
            if which == "receptor":
                if self.focus_mesh_receptor_vao is not None:
                    self.focus_mesh_receptor_vao.release()
                self.focus_mesh_receptor_buffer = buffer
                self.focus_mesh_receptor_vao = vao
                self._focus_mesh_receptor_capacity = capacity
            else:
                if self.focus_mesh_ligand_vao is not None:
                    self.focus_mesh_ligand_vao.release()
                self.focus_mesh_ligand_buffer = buffer
                self.focus_mesh_ligand_vao = vao
                self._focus_mesh_ligand_capacity = capacity
        assert buffer is not None
        buffer.write(flat.tobytes())

    def _focus_geometry(self, which: str) -> None:
        """Build the ball-and-stick geometry of the focused subset."""
        if which == "receptor":
            atoms = list(self.scene.receptor)
            indices = [i for i in self._focus_receptor if 0 <= i < len(atoms)]
            subset = [atoms[i] for i in indices]
            bonds = self.scene.receptor_bonds
            sides = RECEPTOR_STICK_SIDES
            position = {index: local for local, index in enumerate(indices)}
        else:
            atoms = list(self.scene.ligand)
            indices = [i for i in self._focus_ligand if 0 <= i < len(atoms)]
            subset = [atoms[i] for i in indices]
            bonds = self.scene.ligand_bonds
            sides = STICK_SIDES
            position = {index: local for local, index in enumerate(indices)}
        if not subset:
            self._upload_focus_mesh(which, np.zeros((0, 9), dtype="f4"))
            return
        # ``index_map`` maps a global atom index onto the subset position, so a
        # bond with an un-focused end is skipped by build_group_mesh.
        index_map = [position.get(i, -1) for i in range(len(atoms))]
        mesh = build_group_mesh(
            subset,
            bonds,
            "ball_stick",
            index_map=index_map,
            sides=sides,
        )
        self._upload_focus_mesh(which, mesh)

    def upload(self, which: str) -> None:
        if self._sync_focus_from_scene():
            self.dirty_receptor = True
            self.dirty_ligand = True
        if which == "receptor":
            atoms = self.scene.visible_receptor() if self.scene.show_receptor else []
            self._receptor_cached = atoms
            style = canon_protein_style(self.scene.style_protein)
            index_map = None
            if self.scene.receptor_cutoff is not None and self.scene.receptor_radius > 0:
                # A cut-off hides atoms, so a bond with a hidden end must go too.
                lookup = {id(atom): position for position, atom in enumerate(atoms)}
                index_map = [
                    lookup.get(id(atom), -1) for atom in self.scene.receptor
                ]
            mesh = build_group_mesh(
                atoms,
                self.scene.receptor_bonds,
                style,
                index_map=index_map,
                sides=RECEPTOR_STICK_SIDES,
            )
            self._upload_mesh("receptor", mesh)
            self._focus_geometry("receptor")
            if not atoms:
                self.receptor_count = 0
                self.dirty_receptor = False
                return
            data = self._instances_for(atoms, "receptor")
            data = self._without_focused(atoms, data, "receptor")
            self.receptor_count = data.shape[0]
            self._ensure_capacity("receptor", self.receptor_count)
            assert self.receptor_buffer is not None
            self.receptor_buffer.write(data.reshape(-1).tobytes())
            self.dirty_receptor = False
        else:
            atoms = self.scene.ligand if self.scene.show_ligand else []
            style = canon_ligand_style(self.scene.style_ligand)
            mesh = build_group_mesh(atoms, self.scene.ligand_bonds, style)
            self._upload_mesh("ligand", mesh)
            self._focus_geometry("ligand")
            if not atoms:
                self.ligand_count = 0
                self.dirty_ligand = False
                return
            data = self._instances_for(atoms, "ligand")
            data = self._without_focused(atoms, data, "ligand")
            self.ligand_count = data.shape[0]
            self._ensure_capacity("ligand", self.ligand_count)
            assert self.ligand_buffer is not None
            self.ligand_buffer.write(data.reshape(-1).tobytes())
            self.dirty_ligand = False

    # -- appearance --------------------------------------------------------

    def _without_focused(
        self, atoms: Sequence[Atom], data: np.ndarray, which: str
    ) -> np.ndarray:
        """Drop the instance rows of atoms the emphasis already draws itself.

        The focused residues are re-drawn as ball-and-stick with a bright rim;
        leaving their plain spheres in the base pass as well is what buried the
        emphasis behind the protein. With no focus this is a no-op.
        """
        if data.size == 0:
            return data
        suppressed = self._suppressed_atoms(which)
        if not suppressed:
            return data
        keep = np.array([id(atom) not in suppressed for atom in atoms], dtype=bool)
        if keep.size != data.shape[0]:
            return data
        return data[keep]

    def _instances_for(self, atoms: Sequence[Atom], which: str) -> np.ndarray:
        """The instanced *sphere* list for one style of one group.

        The sphere pass and the triangle mesh are complementary halves of a
        style: ball-and-stick is spheres + bond cylinders, the cartoon is a
        mesh with no spheres at all, and ``spheres``/``dots`` are spheres only.
        """
        if which == "receptor":
            style = canon_protein_style(self.scene.style_protein)
            base_scale = self.scene.receptor_scale
        else:
            style = canon_ligand_style(self.scene.style_ligand)
            base_scale = self.scene.ball_scale

        if style == "dots":
            atoms = list(atoms)[:: max(1, len(atoms) // 4000)]
            return self._pack(atoms, base_scale * 0.35)
        if style in ("cartoon", "ribbon"):
            # The mesh carries this style; no spheres at all.
            return np.zeros((0, 7), dtype="f4")
        if style == "tube":
            # The mesh carries this style (``build_tube_mesh``): a real tube, not
            # a chain of spheres whose individual shading reads as beads.
            return np.zeros((0, 7), dtype="f4")
        if style == "wireframe":
            # The skeletal formula: no spheres except the heteroatoms, which are
            # what makes the structure readable without labels.
            entries = []
            for atom in atoms:
                if atom.element in ("C", "H", ""):
                    continue
                r = element_color(atom.element)
                entries.append((atom.x, atom.y, atom.z, 0.17, r[0], r[1], r[2]))
            if not entries:
                return np.zeros((0, 7), dtype="f4")
            return np.asarray(entries, dtype="f4").reshape(-1, 7)
        if style == "sticks":
            # Round joints hide the seam between two bond halves.
            entries = []
            for atom in atoms:
                r = element_color(atom.element)
                entries.append((atom.x, atom.y, atom.z, 0.16, r[0], r[1], r[2]))
            if not entries:
                return np.zeros((0, 7), dtype="f4")
            return np.asarray(entries, dtype="f4").reshape(-1, 7)
        if style == "ball_stick":
            # A ball-and-stick ball is *not* a van der Waals sphere: it has to
            # leave the cylinders visible, so it gets its own (much smaller)
            # scale. ``_pack`` multiplies a van der Waals radius and SPHERE_VS
            # draws the impostor 1.35 x the radius it is handed, so
            # BALL_STICK_SCALE puts a carbon ball at ~0.30 Å on screen — an atom
            # you can see, with the two bond halves still readable under it.
            return self._pack(atoms, BALL_STICK_SCALE)
        if style == "spacefill":
            # Real van der Waals radii at 1:1 — no shrink factor at all — so the
            # ligand is exactly as large as it is and the pocket atoms whose
            # surfaces interpenetrate it are the ones it really touches. This is
            # the style for finding the binding atoms by eye.
            return self._pack(atoms, 1.0)
        # spheres
        return self._pack(atoms, base_scale)

    @staticmethod
    def _pack(atoms: Sequence[Atom], scale: float, radii_already: bool = False) -> np.ndarray:
        """Pack atoms (or prepared tuples) into the instanced vertex format."""
        if radii_already:
            return np.asarray(atoms, dtype="f4").reshape(-1, 7)
        data = np.empty((len(atoms), 7), dtype="f4")
        for i, a in enumerate(atoms):
            data[i, 0] = a.x
            data[i, 1] = a.y
            data[i, 2] = a.z
            data[i, 3] = max(element_radius(a.element) * scale, 0.05)
            data[i, 4:7] = element_color(a.element)
        return data

    def box_handle_positions(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """The six face-centre handles of the search box, as (axis, position)."""
        if self.scene.box is None:
            return []
        center, size = self.scene.box
        out = []
        for axis in range(3):
            for sign in (-1, 1):
                point = list(center)
                point[axis] += sign * max(size[axis], 0.2) / 2.0
                out.append((("x", "y", "z")[axis] + ("-", "+")[sign > 0], tuple(point)))
        return out

    # -- drawing -----------------------------------------------------------

    def reset_gl_state(self, width: int, height: int) -> None:
        """Establish the GL state this renderer relies on.

        The context belongs to this renderer alone — see
        :class:`odock.gui.app.ViewportWidget` for why it is not shared with Qt —
        so this is a plain reset rather than a rescue operation. The
        ``*_direct`` capability calls and the deliberate double writes make it
        correct even if a host ever shares the context: ModernGL caches GL
        state and would otherwise skip a call it believes is already satisfied.

        This matters in practice. ``QOpenGLWidget`` leaves
        ``glColorMask(1, 0, 0, 0)`` and its own blend function behind between
        frames; because ``glClear`` also honours the colour mask, a shared
        context silently discards both the clear and most of the geometry,
        which is what produces a window with an empty 3-D area.
        """
        ctx = self.ctx

        try:
            ctx.disable_direct(moderngl.BLEND)
            ctx.disable_direct(moderngl.CULL_FACE)
            ctx.enable_direct(moderngl.DEPTH_TEST)
        except Exception:  # pragma: no cover - older moderngl
            ctx.disable(moderngl.BLEND)
            ctx.disable(moderngl.CULL_FACE)
            ctx.enable(moderngl.DEPTH_TEST)

        # Writing a different value first guarantees the second write reaches
        # the driver even when ModernGL's cache disagrees.
        ctx.depth_func = ">"
        ctx.depth_func = "<"
        ctx.blend_func = (moderngl.ONE, moderngl.ZERO)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            ctx.color_mask = (False, False, False, False)
            ctx.color_mask = (True, True, True, True)
        except Exception:  # pragma: no cover
            pass
        ctx.scissor = (0, 0, 0, 0)
        ctx.scissor = None
        ctx.viewport = (0, 0, max(int(width), 1), max(int(height), 1))

    def draw(
        self,
        camera: Camera,
        width: int,
        height: int,
        background: Tuple[float, float, float, float] = (0.086, 0.094, 0.125, 1.0),
        target=None,
    ) -> None:
        """Render one frame.

        When ambient occlusion is enabled the scene goes into an internal
        framebuffer first and ``target`` receives the finished image; pass the
        framebuffer you want the result in (the widget does). Without a target
        the scene is drawn straight into whatever is currently bound.
        """
        if self.scene.ssao and target is not None:
            self._ensure_ssao_fbo(width, height)
            self._ssao_fbo.use()
            self._render_scene(camera, width, height, background)
            target.use()
            self._apply_ssao(width, height, camera)
        else:
            self._render_scene(camera, width, height, background)

    def _render_scene(
        self,
        camera: Camera,
        width: int,
        height: int,
        background: Tuple[float, float, float, float],
    ) -> None:
        ctx = self.ctx
        self.reset_gl_state(width, height)
        ctx.clear(*background, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        view = camera.view()
        proj = camera.projection(width, height)
        mvp = (proj @ view).astype("f4")
        self._last_view = view
        self._last_proj = proj

        if self.dirty_receptor:
            self.upload("receptor")
        if self.dirty_ligand:
            self.upload("ligand")

        prog = self.sphere_prog
        # GLSL matrices are column-major while NumPy arrays are row-major,
        # so the transposed buffer is what the shader expects.
        prog["u_view"].write(np.ascontiguousarray(view.T, dtype="f4").tobytes())
        prog["u_proj"].write(np.ascontiguousarray(proj.T, dtype="f4").tobytes())
        # A headlight slightly above and to the left of the camera.
        prog["u_light_view"].value = (0.35, 0.45, 0.82)
        prog["u_alpha"].value = 1.0
        # Front clip, in camera space so it survives orbiting: the plane sits
        # just in front of the ligand's nearest atom, which keeps the whole
        # ligand and cuts away the protein material between it and the camera.
        clip_front = self._front_clip_depth(view)
        self._clip_front = clip_front
        prog["u_clip_front"].value = clip_front
        self.mesh_prog["u_clip_front"].value = clip_front

        # An interaction focus recedes everything that is not part of the site.
        focus = bool(self._focus_receptor or self._focus_ligand)
        dim = float(self.scene.focus_dim) if focus else 0.0
        prog["u_dim"].value = dim
        self.mesh_prog["u_dim"].value = dim
        self.focus_sphere_count = 0

        ctx.enable(moderngl.DEPTH_TEST)
        # The show/hide switches are honoured here as well as at upload time, so
        # a caller may flip the flag and snapshot without re-uploading.
        if self.scene.show_receptor:
            self._draw_mesh("receptor", mvp, view)
            if self.receptor_count and self.receptor_vao is not None:
                self.receptor_vao.render(
                    moderngl.TRIANGLES, vertices=6, instances=self.receptor_count
                )

        # Highlighted atoms (the picked one) are drawn enlarged and white-hot.
        self._draw_highlight(prog)

        if self.scene.show_ligand:
            self._draw_mesh("ligand", mvp, view)
            if self.ligand_count and self.ligand_vao is not None:
                self.ligand_vao.render(
                    moderngl.TRIANGLES, vertices=6, instances=self.ligand_count
                )

        # The interaction site is drawn opaque and un-dimmed, on top of the
        # receded structure: the focused residues as ball-and-stick, their atoms
        # and the ligand's contacting atoms with a bright rim.
        if focus:
            self._draw_focus(mvp, view)

        # The selection is opaque, depth-tested, drawn with the geometry it
        # marks (before the translucent reference so it is never tinted by it).
        self._draw_selection()

        self._draw_interactions(camera, width, height, mvp, view)
        self._draw_axes(camera, mvp)
        self._draw_measurements(mvp)

        # The reference pose is blended on top of the finished scene and drawn
        # before the box: depth testing keeps it behind the protein, and nothing
        # that must stay crisp is drawn after the translucent box fill (which is
        # depth-tested, so it writes depth where it is in front).
        self._draw_ghost()

        # The search box goes last and *depth-tested*: atoms closer to the camera
        # than the box's near face occlude the fill, so the fill can only ever
        # tint what is behind it. Drawing it with the depth test off (as it was)
        # let it paint over the whole viewport and made the receptor look
        # invisible as soon as a ligand was loaded and the camera zoomed in.
        self._draw_box(camera, mvp)

    # -- annotation passes --------------------------------------------------

    #: Colour of a selected atom (bright amber) and how much bigger than the
    #: style's own sphere it is drawn. The boost is deliberately small: under
    #: ``spacefill`` a large factor would turn a selection into a wall of amber
    #: spheres, so the *colour* does the work and the radius only adds a rim.
    SELECTION_COLOR = (1.00, 0.94, 0.36)
    SELECTION_BOOST = 1.12
    SELECTION_PADDING = 0.05

    def _atom_sphere_radius(self, atom, which: str) -> float:
        """The sphere radius the current style draws ``atom`` with.

        Used by the selection pass so a highlighted atom is exactly a little
        larger than the ball it covers, whatever the style is — including the
        styles (cartoon, ribbon, tube) that draw no atom spheres at all, where
        a small fallback ball is the only thing that can mark the atom.
        """
        if which == "receptor":
            style = canon_protein_style(self.scene.style_protein)
            scale = self.scene.receptor_scale
        else:
            style = canon_ligand_style(self.scene.style_ligand)
            scale = self.scene.ball_scale
        if style == "spacefill":
            factor = 1.0
        elif style in ("spheres", "dots"):
            factor = scale
        elif style == "ball_stick":
            factor = BALL_STICK_SCALE
        else:
            return max(0.22, element_radius(atom.element) * 0.35)
        return element_radius(atom.element) * factor

    def _overlay_draw(self, which: str, entries, alpha: float, clip_front: float = 0.0) -> int:
        """Upload and draw one overlay instance list; returns its instance count.

        The draw uses whatever GL state the frame already has (depth test on,
        blending on), so both overlays are correctly occluded by the scene.
        ``clip_front`` reuses the scene's front cut-away: a selected atom the
        scene cut away must vanish too, otherwise the highlight would draw the
        very material the cut-away removed and block the view again.
        """
        count = len(entries)
        if which == "selection":
            self.selection_count = count
        else:
            self.ghost_count = count
        if not count:
            return 0
        data = np.asarray(entries, dtype="f4").reshape(-1, 7)
        buffer = self._overlay_buffers.get(which)
        if buffer is None or count > self._overlay_capacity.get(which, 0):
            if buffer is not None:
                buffer.release()
            old_vao = self._overlay_vaos.get(which)
            if old_vao is not None:
                old_vao.release()
            capacity = int(2 ** max(6, math.ceil(math.log2(max(count, 1)))))
            buffer = self.ctx.buffer(reserve=28 * capacity, dynamic=True)
            vao = self.ctx.vertex_array(
                self.overlay_prog,
                [
                    (self.quad_buffer, "2f", "in_quad"),
                    (buffer, "3f 4f/i", "in_center", "in_radius_color"),
                ],
            )
            self._overlay_buffers[which] = buffer
            self._overlay_vaos[which] = vao
            self._overlay_capacity[which] = capacity
        buffer.write(data.reshape(-1).tobytes())

        program = self.overlay_prog
        program["u_view"].write(
            np.ascontiguousarray(self._last_view.T, dtype="f4").tobytes()
        )
        program["u_proj"].write(
            np.ascontiguousarray(self._last_proj.T, dtype="f4").tobytes()
        )
        program["u_light_view"].value = (0.35, 0.45, 0.82)
        program["u_alpha"].value = float(alpha)
        # An overlay is never part of the dimmed mass: the selection and the
        # emphasised site must read at full strength.
        program["u_dim"].value = 0.0
        program["u_clip_front"].value = float(clip_front)
        self._overlay_vaos[which].render(
            moderngl.TRIANGLES, vertices=6, instances=count
        )
        return count

    def _draw_focus(self, mvp: np.ndarray, view: np.ndarray) -> None:
        """Draw the emphasised interaction site, un-dimmed and on top.

        Two halves: the focused residues' ball-and-stick mesh (built on upload,
        so toggling is cheap) and a bright rim of spheres over every focused
        atom — covering both the atoms the interaction names and the rest of
        their residues. The contact dashes are drawn after this pass, so the
        geometry of the interaction is never buried under the emphasis.
        """
        program = self.mesh_prog
        program["u_dim"].value = 0.0
        program["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
        program["u_view"].write(np.ascontiguousarray(view.T, dtype="f4").tobytes())
        program["u_light_view"].value = (0.35, 0.45, 0.82)
        program["u_alpha"].value = 1.0
        for which in ("receptor", "ligand"):
            if which == "receptor":
                vertices = self.focus_mesh_receptor_vertices
                vao = self.focus_mesh_receptor_vao
            else:
                vertices = self.focus_mesh_ligand_vertices
                vao = self.focus_mesh_ligand_vao
            if vertices and vao is not None:
                vao.render(moderngl.TRIANGLES, vertices=vertices)

        entries = []
        key_entries = []
        for which, indices, keys, atoms in (
            (
                "receptor",
                self._focus_receptor,
                set(self._focus_key_receptor),
                self.scene.receptor,
            ),
            (
                "ligand",
                self._focus_ligand,
                set(self._focus_key_ligand),
                self.scene.ligand,
            ),
        ):
            if which == "receptor" and not self.scene.show_receptor:
                continue
            if which == "ligand" and not self.scene.show_ligand:
                continue
            for index in indices:
                if not (0 <= index < len(atoms)):
                    continue
                atom = atoms[index]
                base = element_radius(atom.element) * BALL_STICK_SCALE
                if index in keys:
                    # The two atoms the contact names: the hottest thing in the
                    # frame apart from the dash itself.
                    radius = base * FOCUS_KEY_BOOST + FOCUS_PADDING
                    key_entries.append(
                        (atom.x, atom.y, atom.z, radius, *focus_key_color(atom.element))
                    )
                else:
                    radius = base * FOCUS_BOOST + FOCUS_PADDING
                    entries.append(
                        (atom.x, atom.y, atom.z, radius, *focus_color(atom.element))
                    )
        # The named pair goes last so it is drawn over the rest of its residue.
        self.focus_sphere_count = self._overlay_draw(
            "focus", entries + key_entries, 1.0, self._clip_front
        )

    def _draw_selection(self) -> None:
        """The selected atoms, bright and slightly enlarged, depth-tested.

        ``scene.selection`` is ``[("receptor" | "ligand", index), ...]``; an
        empty list is a no-op, and an index outside its structure is skipped
        rather than raising.
        """
        selection = self.scene.selection or []
        if not selection:
            self.selection_count = 0
            return
        entries = []
        for item in selection:
            try:
                which = str(item[0])
                index = int(item[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if which == "receptor":
                if not self.scene.show_receptor:
                    continue
                atoms = self.scene.receptor
            elif which == "ligand":
                if not self.scene.show_ligand:
                    continue
                atoms = self.scene.ligand
            else:
                continue
            if not (0 <= index < len(atoms)):
                continue
            atom = atoms[index]
            radius = (
                self._atom_sphere_radius(atom, which) * self.SELECTION_BOOST
                + self.SELECTION_PADDING
            )
            entries.append((atom.x, atom.y, atom.z, radius, *self.SELECTION_COLOR))
        self._overlay_draw(
            "selection", entries, 1.0, getattr(self, "_clip_front", 0.0)
        )

    def _draw_ghost(self) -> None:
        """The reference ligand as a translucent cloud behind the current pose.

        Drawn with the sphere program at ``scene.ghost_alpha``, depth-tested and
        last in the frame, so the protein in front of it hides it while the
        current pose stays crisp inside it.
        """
        ghost = self.scene.ghost_ligand or []
        if not ghost or not self.scene.show_ligand:
            self.ghost_count = 0
            return
        scale = max(0.05, float(self.scene.ghost_scale))
        entries = []
        for atom in ghost:
            radius = max(self._atom_sphere_radius(atom, "ligand"), 0.35) * scale
            color = element_color(atom.element)
            entries.append((atom.x, atom.y, atom.z, radius, *color))
        self._overlay_draw("ghost", entries, float(self.scene.ghost_alpha))

    def _front_clip_depth(self, view: np.ndarray) -> float:
        """The camera-space depth of the front clip plane (0 = no clipping).

        A ligand bound in a pocket is surrounded by protein, so nothing of it is
        visible from outside. The plane is placed 0.6 Å in front of the ligand's
        nearest atom, in *camera* space, which means the cut-away rotates with
        the model and the ligand is never clipped away itself.
        """
        if not self.scene.front_clip:
            return 0.0
        atoms = self.scene.ligand if self.scene.show_ligand else []
        if not atoms:
            return 0.0
        matrix = np.asarray(view, dtype="f8")
        nearest = None
        for atom in atoms:
            point = matrix @ np.array([atom.x, atom.y, atom.z, 1.0])
            depth = -float(point[2])
            if nearest is None or depth < nearest:
                nearest = depth
        if nearest is None:  # pragma: no cover - defensive
            return 0.0
        return max(0.0, nearest - 0.6)

    def _draw_mesh(self, which: str, mvp: np.ndarray, view: np.ndarray) -> None:
        """One triangle draw for the group's bond cylinders / cartoon / ribbon."""
        if which == "receptor":
            vertices = self.mesh_receptor_vertices
            vao = self.mesh_receptor_vao
        else:
            vertices = self.mesh_ligand_vertices
            vao = self.mesh_ligand_vao
        if not vertices or vao is None:
            return
        program = self.mesh_prog
        program["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
        program["u_view"].write(np.ascontiguousarray(view.T, dtype="f4").tobytes())
        program["u_light_view"].value = (0.35, 0.45, 0.82)
        program["u_alpha"].value = 1.0
        vao.render(moderngl.TRIANGLES, vertices=vertices)

    def _draw_highlight(self, prog) -> None:
        index = self.scene.highlight
        if not index or not self.scene.show_ligand:
            return
        atoms = self.scene.ligand
        entries = []
        for i in index:
            if 0 <= i < len(atoms):
                a = atoms[i]
                entries.append((a.x, a.y, a.z, element_radius(a.element) * 0.9, 1.0, 0.85, 0.2))
        if not entries:
            return
        data = np.asarray(entries, dtype="f4")
        self.handle_buffer.write(data.tobytes())
        self.handle_prog["u_view"].write(
            np.ascontiguousarray(self._last_view.T, dtype="f4").tobytes()
        )
        self.handle_prog["u_proj"].write(
            np.ascontiguousarray(self._last_proj.T, dtype="f4").tobytes()
        )
        self.handle_prog["u_light_view"].value = (0.35, 0.45, 0.82)
        self.handle_prog["u_alpha"].value = 1.0
        self.handle_vao.render(moderngl.TRIANGLES, vertices=6, instances=len(entries))

    #: Dash geometry: a dashed line is just many short segments.
    DASH_LENGTH = 0.28
    DASH_GAP = 0.20

    def _dashed(self, a, b, color, width: float = 1.0) -> List[tuple]:
        """A dashed segment as a list of (x1,y1,z1,r,g,b,a, x2,y2,z2,r,g,b,a)."""
        a = np.asarray(a, dtype="f8")
        b = np.asarray(b, dtype="f8")
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return []
        direction = delta / length
        out = []
        t = 0.0
        while t < length:
            t0 = t
            t1 = min(t + self.DASH_LENGTH, length)
            p0 = a + direction * t0
            p1 = a + direction * t1
            out.append((*p0, *color, *p1, *color))
            t = t1 + self.DASH_GAP
        return out

    #: Target on-screen width of a contact dash, in pixels. A GL line is one
    #: pixel wide whatever the hardware, which is invisible against a protein, so
    #: each dash is swept as a cylinder whose radius is chosen per segment for
    #: this width (see :meth:`_draw_interaction_tubes`).
    INTERACTION_PIXELS = 3.0
    #: Radial segments of a dash tube: a handful is plenty at 3 px.
    INTERACTION_SIDES = 5

    def _draw_interactions(
        self,
        camera: Camera,
        width: int,
        height: int,
        mvp: np.ndarray,
        view: np.ndarray,
    ) -> None:
        """The contact annotations, as fat coloured tubes rather than hairlines."""
        interactions = self.scene.interactions
        if not interactions:
            self.interaction_vertices = 0
            return
        receptor = self.scene.receptor
        ligand = self.scene.ligand
        segments: List[tuple] = []
        for item in interactions:
            kind = getattr(item, "kind", "")
            a = getattr(item, "a", None)
            b = getattr(item, "b", None)
            if a is None or b is None:
                continue
            if not (0 <= a < len(receptor) and 0 <= b < len(ligand)):
                continue
            color = INTERACTION_COLORS.get(kind, (0.8, 0.8, 0.8, 0.9))
            ra = receptor[a]
            rb = ligand[b]
            pa = (ra.x, ra.y, ra.z)
            pb = (rb.x, rb.y, rb.z)
            if kind == "hydrophobic":
                # A hydrophobic contact is a contact, not a directional bond:
                # draw it as a sparse dotted column instead of a dashed line.
                segments.extend(self._dotted(pa, pb, color, step=0.55))
            else:
                segments.extend(self._dashed(pa, pb, color))
        self._draw_interaction_tubes(segments, mvp, view, height)

    def _draw_interaction_tubes(
        self, segments, mvp: np.ndarray, view: np.ndarray, height: int
    ) -> None:
        """Sweep the contact dashes as screen-space-width cylinders.

        The radius of each dash is derived from its distance to the camera so the
        result is ``INTERACTION_PIXELS`` wide on screen: the world height covered
        by one pixel at depth ``d`` is ``2 d / (f * height)``, with ``f =
        u_proj[1][1] = 1/tan(fov/2)``.
        """
        if not segments:
            self.interaction_vertices = 0
            return
        projection = np.asarray(self._last_proj, dtype="f8")
        if abs(projection[1, 1]) < 1e-9:  # pragma: no cover - degenerate camera
            return
        matrix = np.asarray(view, dtype="f8")
        starts: List[np.ndarray] = []
        ends: List[np.ndarray] = []
        radii: List[float] = []
        colors: List[np.ndarray] = []
        for segment in segments:
            start = np.asarray(segment[0:3], dtype="f8")
            end = np.asarray(segment[6:9], dtype="f8")
            colour = np.asarray(segment[3:6], dtype="f8")
            middle = np.append(0.5 * (start + end), 1.0)
            depth = -float((matrix @ middle)[2])
            if depth <= 1e-4:
                continue  # behind the camera
            world_per_pixel = 2.0 * depth / (projection[1, 1] * max(height, 1))
            radius = 0.5 * self.INTERACTION_PIXELS * world_per_pixel
            starts.append(start)
            ends.append(end)
            radii.append(min(0.6, max(0.008, radius)))
            colors.append(colour)
        if not starts:
            self.interaction_vertices = 0
            return
        mesh = _cylinders(
            np.asarray(starts),
            np.asarray(ends),
            np.asarray(radii),
            np.asarray(colors),
            sides=self.INTERACTION_SIDES,
            caps=True,
        )
        self._upload_interaction_mesh(mesh)
        if not self.interaction_vertices or self.interaction_vao is None:
            return
        program = self.mesh_prog
        program["u_dim"].value = 0.0
        program["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
        program["u_view"].write(np.ascontiguousarray(view.T, dtype="f4").tobytes())
        program["u_light_view"].value = (0.35, 0.45, 0.82)
        program["u_alpha"].value = 1.0
        self.interaction_vao.render(
            moderngl.TRIANGLES, vertices=self.interaction_vertices
        )

    def _upload_interaction_mesh(self, data: np.ndarray) -> None:
        """Upload the dash tubes into their own mesh buffer."""
        vertices = 0 if data is None or data.size == 0 else int(data.shape[0])
        self.interaction_vertices = vertices
        if vertices == 0:
            return
        flat = np.ascontiguousarray(data.reshape(-1), dtype="f4")
        if self.interaction_buffer is None or flat.size > self._interaction_capacity:
            if self.interaction_buffer is not None:
                self.interaction_buffer.release()
            if self.interaction_vao is not None:
                self.interaction_vao.release()
            capacity = max(flat.size, 9 * 4096)
            self.interaction_buffer = self.ctx.buffer(
                reserve=int(capacity * 4), dynamic=True
            )
            self.interaction_vao = self.ctx.vertex_array(
                self.mesh_prog,
                [(self.interaction_buffer, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
            )
            self._interaction_capacity = capacity
        assert self.interaction_buffer is not None
        self.interaction_buffer.write(flat.tobytes())


    def _dotted(self, a, b, color, step: float = 0.5) -> List[tuple]:
        a = np.asarray(a, dtype="f8")
        b = np.asarray(b, dtype="f8")
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return []
        direction = delta / length
        out = []
        t = step * 0.5
        while t < length:
            p = a + direction * t
            p2 = p + direction * (step * 0.25)
            out.append((*p, *color, *p2, *color))
            t += step
        return out

    def _draw_measurements(self, mvp: np.ndarray) -> None:
        segments = []
        for item in self.scene.measurements:
            try:
                a, b = item[0], item[1]
                color = item[2] if len(item) > 2 else (1.0, 0.85, 0.3, 1.0)
            except Exception:  # pragma: no cover - defensive
                continue
            segments.append((*a, *color, *b, *color))
        self._draw_lines(segments, mvp)

    def _draw_axes(self, camera: Camera, mvp: np.ndarray) -> None:
        if not self.scene.show_axes:
            return
        eye = np.asarray(camera.eye(), dtype="f8")
        distance = max(camera.distance * 0.18, 3.0)
        origin = eye
        segments = []
        for color, direction in (
            ((0.95, 0.35, 0.35, 1.0), (distance, 0, 0)),
            ((0.45, 0.9, 0.45, 1.0), (0, distance, 0)),
            ((0.45, 0.6, 1.0, 1.0), (0, 0, distance)),
        ):
            tip = origin + np.asarray(direction, dtype="f8")
            segments.append((*origin, *color, *tip, *color))
        self._draw_lines(segments, mvp)

    def _draw_lines(self, segments, mvp: np.ndarray) -> None:
        if not segments:
            return
        data = np.asarray(segments, dtype="f4").reshape(-1)
        needed = len(segments) * 2
        if needed > self._line_capacity:
            capacity = int(2 ** max(10, math.ceil(math.log2(needed))))
            if self.line_vao is not None:
                self.line_vao.release()
            if self.line_buffer is not None:
                self.line_buffer.release()
            self.line_buffer = self.ctx.buffer(reserve=int(capacity * 28), dynamic=True)
            self.line_vao = self.ctx.vertex_array(
                self.line_prog,
                [(self.line_buffer, "3f 4f", "in_pos", "in_color")],
            )
            self._line_capacity = capacity
        assert self.line_buffer is not None and self.line_vao is not None
        self.line_buffer.write(data.tobytes())
        self.line_prog["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
        self.line_vao.render(moderngl.LINES, vertices=needed)

    # -- ambient occlusion --------------------------------------------------

    def _ensure_ssao_fbo(self, width: int, height: int) -> None:
        if self._ssao_fbo is not None and self._ssao_size == (width, height):
            return
        if self._ssao_fbo is not None:
            self._ssao_fbo.release()
        color = self.ctx.texture((width, height), 3)
        depth = self.ctx.depth_texture((width, height))
        self._ssao_fbo = self.ctx.framebuffer(
            color_attachments=[color], depth_attachment=depth
        )
        self._ssao_size = (width, height)

    def _apply_ssao(self, width: int, height: int, camera: Camera) -> None:
        ctx = self.ctx
        assert self._ssao_fbo is not None
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.viewport = (0, 0, width, height)
        self._ssao_fbo.color_attachments[0].use(0)
        self._ssao_fbo.depth_attachment.use(1)
        near = max(0.05, camera.distance * 0.005)
        far = camera.distance * 20.0 + 500.0
        self.fs_prog["u_color"].value = 0
        self.fs_prog["u_depth"].value = 1
        self.fs_prog["u_texel"].value = (1.0 / max(width, 1), 1.0 / max(height, 1))
        self.fs_prog["u_near"].value = float(near)
        self.fs_prog["u_far"].value = float(far)
        self.fs_prog["u_strength"].value = float(self.scene.ssao_strength)
        self.fs_vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)

    def _draw_box(self, camera: Camera, mvp: np.ndarray) -> None:
        """The search box: a translucent fill plus its edges.

        The fill is what makes "which region is the search space" readable at a
        glance; the edges stay visible even at ``box_alpha == 0`` so the box is
        never invisible. The six face grips are drawn only when
        ``Scene.show_box_handles`` asks for them.
        """
        if self.scene.box is None:
            self.box_fill_triangles = 0
            self.box_edge_vertices = 0
            self.handle_count = 0
            self.handle_positions = []
            return
        center, size = self.scene.box
        scale = np.array(
            [max(size[0], 0.2) / 2.0, max(size[1], 0.2) / 2.0, max(size[2], 0.2) / 2.0],
            dtype="f8",
        )
        corners = self.box_corners.astype("f8") * scale + np.array(center, dtype="f8")
        alpha = min(1.0, max(0.0, float(self.scene.box_alpha)))
        self._draw_box_fill(center, scale, camera, alpha, mvp)

        self.box_buffer.write(corners.astype("f4").tobytes())
        self.box_prog["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
        # The edges keep a floor of their own alpha so a faint fill, or none at
        # all, still outlines the box.
        self.box_prog["u_color"].value = (0.30, 0.85, 0.95, min(1.0, alpha + 0.45))
        self.box_vao.render(moderngl.LINES)
        self.box_edge_vertices = 24
        if self.scene.show_box_handles:
            self._draw_box_handles()
        else:
            self.handle_count = 0
            self.handle_positions = []

    def _draw_box_fill(
        self,
        center,
        half,
        camera: Camera,
        alpha: float,
        mvp: np.ndarray,
    ) -> None:
        """Blend the box's translucent volume, one layer thick, depth-tested.

        Only the faces turned towards the camera are emitted (all six would
        blend six layers of the same alpha, which reads as opaque), and the pass
        keeps the depth test **on** so an atom closer to the camera than the
        box's near face occludes the fill. That is what keeps the box from
        washing out the structure: the fill can only tint what is actually
        behind it. Nothing that must stay crisp is drawn after this pass (it is
        last in the frame apart from nothing at all), so the depth it writes
        where it is in front cannot hide anything either.

        When the camera is *inside* the box the fill is skipped entirely: those
        faces are behind the eye and every remaining one projects over the whole
        viewport. The edges are still drawn, so the box stays outlined.
        """
        self.box_fill_triangles = 0
        if alpha <= 0.001:
            return
        centre = np.asarray(center, dtype="f8")
        extent = np.asarray(half, dtype="f8")
        eye = np.asarray(camera.eye(), dtype="f8")
        # Inside test in box space: the offset from the centre, per axis, against
        # the box's own half extents.
        if bool(np.all(np.abs(eye - centre) <= extent)):
            return
        to_eye = eye - centre
        # The eight corners in the same order as ``Renderer.box_corners``.
        signs = np.array(
            [
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            ],
            dtype="f8",
        )
        corners = centre + signs * extent
        vertices: List[np.ndarray] = []
        for quad, normal in _BOX_FACES:
            if float(np.dot(np.asarray(normal, dtype="f8"), to_eye)) <= 0.0:
                continue
            a, b, c, d = (corners[index] for index in quad)
            vertices.extend((a, b, c, a, c, d))
        if not vertices:
            return
        data = np.asarray(vertices, dtype="f4")
        self.box_fill_buffer.write(data.tobytes())
        self.box_prog["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
        self.box_prog["u_color"].value = (0.30, 0.85, 0.95, alpha)
        self.box_fill_vao.render(moderngl.TRIANGLES, vertices=len(data))
        self.box_fill_triangles = len(data) // 3

    def _draw_box_handles(self) -> None:
        """The six draggable face handles, shown as small orange spheres.

        They are drawn with the sphere program so they are shaded and
        depth-tested like everything else, which is what makes them feel like
        part of the scene rather than an overlay.
        """
        handles = self.box_handle_positions()
        self.handle_positions = [position for _, position in handles]
        if not handles:
            self.handle_count = 0
            return
        span = max(self.scene.box[1]) if self.scene.box else 10.0
        radius = max(0.12, min(5.0, span * 0.035))
        entries = [(x, y, z, radius, 1.0, 0.62, 0.18) for _, (x, y, z) in handles]
        data = np.asarray(entries, dtype="f4")
        self.handle_buffer.write(data.tobytes())
        self.handle_prog["u_view"].write(
            np.ascontiguousarray(self._last_view.T, dtype="f4").tobytes()
        )
        self.handle_prog["u_proj"].write(
            np.ascontiguousarray(self._last_proj.T, dtype="f4").tobytes()
        )
        self.handle_prog["u_light_view"].value = (0.35, 0.45, 0.82)
        self.handle_prog["u_alpha"].value = 1.0
        self.handle_vao.render(moderngl.TRIANGLES, vertices=6, instances=len(entries))
        self.handle_count = len(entries)

    # -- picking ------------------------------------------------------------

    def screen_ray(self, camera: Camera, width: int, height: int, x: float, y: float):
        """Unproject a widget pixel into a world-space ray (origin, direction)."""
        view = camera.view()
        proj = camera.projection(width, height)
        inverse = np.linalg.inv((proj @ view).astype("f8"))
        ndc_x = 2.0 * x / max(width, 1) - 1.0
        ndc_y = 1.0 - 2.0 * y / max(height, 1)
        near = inverse @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far = inverse @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        near = near[:3] / near[3]
        far = far[:3] / far[3]
        direction = far - near
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:  # pragma: no cover - degenerate camera
            return near, np.array([0.0, 0.0, -1.0])
        return near, direction / norm

    def pick_atom(
        self,
        camera: Camera,
        width: int,
        height: int,
        x: float,
        y: float,
        *,
        ligand_first: bool = True,
    ):
        """Return ``('ligand'|'receptor', index)`` under the cursor, or ``None``.

        A plain ray/sphere test: the nearest atom whose sphere the ray enters
        wins. That is exact for the sphere style, which is what the user sees.
        """
        origin, direction = self.screen_ray(camera, width, height, x, y)
        best = None
        best_t = float("inf")
        # ``visible_receptor()`` is a *filtered* list, so a hit has to be mapped
        # back onto an index into ``scene.receptor``: without this every pick
        # while a binding-site cut-off is active is off by the atoms the cut-off
        # removed (the measure tool resolved the wrong atom, and a click selected
        # the wrong residue).
        visible = (
            self.scene.visible_receptor() if self.scene.show_receptor else []
        )
        receptor_map = None
        if self.scene.receptor_cutoff is not None and self.scene.receptor_radius > 0:
            receptor_map = {id(atom): index for index, atom in enumerate(self.scene.receptor)}
        receptor_group = ("receptor", visible, self.scene.receptor_scale, receptor_map)
        ligand_group = ("ligand", self.scene.ligand, self.scene.ball_scale, None)
        groups = (
            [ligand_group, receptor_group]
            if ligand_first
            else [receptor_group, ligand_group]
        )
        for kind, atoms, scale, remap in groups:
            for index, atom in enumerate(atoms):
                radius = element_radius(atom.element) * scale
                if radius <= 0:
                    continue
                to_center = np.array([atom.x, atom.y, atom.z]) - origin
                projection = float(np.dot(to_center, direction))
                if projection < 0:
                    continue
                perpendicular = float(np.dot(to_center, to_center)) - projection * projection
                if perpendicular > radius * radius:
                    continue
                if projection < best_t:
                    best_t = projection
                    best = (kind, remap.get(id(atom), index) if remap else index)
        return best

    def pick_box_handle(
        self, camera: Camera, width: int, height: int, x: float, y: float
    ):
        """Return the index of the box handle under the cursor, or ``None``."""
        handles = self.box_handle_positions()
        if not handles:
            return None
        origin, direction = self.screen_ray(camera, width, height, x, y)
        span = max(self.scene.box[1]) if self.scene.box else 10.0
        radius = max(0.12, min(5.0, span * 0.035)) * 2.2  # a generous hit area
        best_index, best_t = None, float("inf")
        for index, (_, position) in enumerate(handles):
            to_center = np.asarray(position, dtype="f8") - origin
            projection = float(np.dot(to_center, direction))
            if projection < 0:
                continue
            perpendicular = float(np.dot(to_center, to_center)) - projection * projection
            if perpendicular <= radius * radius and projection < best_t:
                best_t, best_index = projection, index
        return best_index

    # -- offscreen rendering (high-resolution screenshots) ------------------

    def render_image(self, camera: Camera, width: int, height: int, background=(0.086, 0.094, 0.125, 1.0)):
        """Render one frame at an arbitrary size and return it as RGB bytes."""
        ctx = self.ctx
        color = ctx.texture((width, height), 3)
        depth = ctx.depth_texture((width, height))
        fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        previous_ssao = self.scene.ssao
        try:
            fbo.use()
            # SSAO allocates its own framebuffer of the same size; keep it on.
            self.scene.ssao = previous_ssao
            self._render_scene(camera, width, height, background)
            if previous_ssao:
                target = fbo
                self._ensure_ssao_fbo(width, height)
                self._ssao_fbo.use()
                self._render_scene(camera, width, height, background)
                target.use()
                self._apply_ssao(width, height, camera)
            ctx.finish()
            return fbo.read(components=3)
        finally:
            self.scene.ssao = previous_ssao
            fbo.release()

    def release(self) -> None:
        for obj in (
            self.receptor_vao,
            self.ligand_vao,
            self.box_vao,
            self.line_vao,
            self.handle_vao,
            self.fs_vao,
            self.mesh_receptor_vao,
            self.mesh_ligand_vao,
            self.focus_mesh_receptor_vao,
            self.focus_mesh_ligand_vao,
            self.interaction_vao,
            self.box_fill_vao,
            self.quad_buffer,
            self.receptor_buffer,
            self.ligand_buffer,
            self.box_buffer,
            self.box_index_buffer,
            self.line_buffer,
            self.handle_buffer,
            self.fs_quad,
            self.mesh_receptor_buffer,
            self.mesh_ligand_buffer,
            self.focus_mesh_receptor_buffer,
            self.focus_mesh_ligand_buffer,
            self.interaction_buffer,
            self.box_fill_buffer,
            self.sphere_prog,
            self.box_prog,
            self.line_prog,
            self.handle_prog,
            self.fs_prog,
            self.mesh_prog,
            self.overlay_prog,
            *self._overlay_buffers.values(),
            *self._overlay_vaos.values(),
        ):
            if obj is None:
                continue
            try:
                obj.release()
            except Exception:
                pass
        if self._ssao_fbo is not None:
            try:
                self._ssao_fbo.release()
            except Exception:  # pragma: no cover
                pass
            self._ssao_fbo = None
