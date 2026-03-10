#version 460
//-----------------------------------------------------------------------------
// Program Type: Fragment shader
// Language: glsl
// Created by Ogre RT Shader Generator. All rights reserved.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//                         FORWARD DECLARATIONS
//-----------------------------------------------------------------------------
void FFP_Add(in vec3, in vec3, out vec3);
void FFP_Assign(in vec3, out vec3);
void FFP_Assign(in vec4, out vec4);
void FFP_Construct(in float, in float, in float, in float, out vec4);
void FFP_Modulate(in float, in float, out float);
void FFP_SampleTexture(in sampler2D, in vec2, out vec4);
void SGX_ApplyShadowFactor_Diffuse(in vec4, in vec4, in float, out vec4);
void SGX_ComputeShadowFactor_PSSM3(in float, in vec4, in vec4, in vec4, in vec4, in sampler2D, in sampler2D, in sampler2D, in vec4, in vec4, in vec4, out float);
void SGX_Light_Directional_DiffuseSpecular(in vec3, in vec3, in vec3, in vec3, in vec3, in float, in vec3, in vec3, out vec3, out vec3);
void SGX_ModulateScalar(in float, in vec4, out vec4);

//-----------------------------------------------------------------------------
//                         GLOBAL PARAMETERS
//-----------------------------------------------------------------------------

uniform	vec4	derived_ambient_light_colour;
uniform	vec4	surface_diffuse_colour;
uniform	vec4	surface_specular_colour;
uniform	vec4	surface_emissive_colour;
uniform	vec4	derived_scene_colour;
uniform	float	surface_shininess;
uniform	vec4	light_direction_view_space0;
uniform	vec4	derived_light_diffuse1;
uniform	vec4	derived_light_specular2;
uniform	sampler2D	gTextureSampler0;
uniform	vec4	pssm_split_points3;
uniform	sampler2D	shadow_map1;
uniform	vec4	inv_shadow_texture_size4;
uniform	sampler2D	shadow_map2;
uniform	vec4	inv_shadow_texture_size5;
uniform	sampler2D	shadow_map3;
uniform	vec4	inv_shadow_texture_size6;

//-----------------------------------------------------------------------------
// Function Name: main
// Function Desc: Pixel Program Entry point
//-----------------------------------------------------------------------------
in	vec3	oTexcoord3_0;
in	vec3	oTexcoord3_1;
in	vec2	oTexcoord2_2;
in	float	oTexcoord1_3;
in	vec4	oTexcoord4_4;
in	vec4	oTexcoord4_5;
in	vec4	oTexcoord4_6;
out vec4 fragColour;
void main(void) {
	vec4	lLocalParam_0;
	vec4	lLocalParam_1;
	vec4	lPerPixelDiffuse;
	vec4	lPerPixelSpecular;
	vec4	texel_0;
	vec4	source1;
	vec4	source2;
	float	lShadowFactor;

	FFP_Construct(1.0, 1.0, 1.0, 1.0, lLocalParam_0);

	FFP_Construct(0.0, 0.0, 0.0, 0.0, lLocalParam_1);

	FFP_Assign(lLocalParam_0, fragColour);

	FFP_Assign(derived_scene_colour, lPerPixelDiffuse);

	FFP_Assign(lLocalParam_1, lPerPixelSpecular);

	SGX_Light_Directional_DiffuseSpecular(oTexcoord3_0, oTexcoord3_1, light_direction_view_space0.xyz, derived_light_diffuse1.xyz, derived_light_specular2.xyz, surface_shininess, lPerPixelDiffuse.xyz, lPerPixelSpecular.xyz, lPerPixelDiffuse.xyz, lPerPixelSpecular.xyz);

	FFP_Assign(lPerPixelDiffuse, lLocalParam_0);

	FFP_Assign(lLocalParam_0, fragColour);

	FFP_Assign(lPerPixelSpecular, lLocalParam_1);

	SGX_ComputeShadowFactor_PSSM3(oTexcoord1_3, pssm_split_points3, oTexcoord4_4, oTexcoord4_5, oTexcoord4_6, shadow_map1, shadow_map2, shadow_map3, inv_shadow_texture_size4, inv_shadow_texture_size5, inv_shadow_texture_size6, lShadowFactor);

	SGX_ApplyShadowFactor_Diffuse(derived_scene_colour, lLocalParam_0, lShadowFactor, lLocalParam_0);

	SGX_ModulateScalar(lShadowFactor, lLocalParam_1, lLocalParam_1);

	FFP_Assign(lLocalParam_0, fragColour);

	FFP_SampleTexture(gTextureSampler0, oTexcoord2_2, texel_0);

	FFP_Assign(lLocalParam_0, source1);

	FFP_Assign(texel_0, source2);

	FFP_Assign(source1.xyz, fragColour.xyz);

	FFP_Assign(texel_0, source1);

	FFP_Assign(lLocalParam_0, source2);

	FFP_Modulate(source1.w, source2.w, fragColour.w);

	FFP_Add(fragColour.xyz, lLocalParam_1.xyz, fragColour.xyz);

}

