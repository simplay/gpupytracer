#version 430
uniform sampler2D tex;
in vec2 v_uv;
out vec4 color;
void main() {
    vec3 raw = texture(tex, v_uv).rgb;
    color = vec4(pow(raw, vec3(1.0 / 2.2)), 1.0);
}