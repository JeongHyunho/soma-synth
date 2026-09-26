"""One-line semantics for every field of the qmd_unified8_smpl18 dataset spec.

The spec module owns structure (dtype/shape/axes, CORE vs capability-conditional); this registry owns the
prose. The README generator merges the two with live introspection. A completeness test asserts every spec
field/enum has an entry here, so a new field cannot ship undocumented.
"""

from __future__ import annotations

ARTIFACT_DOCS: dict[str, str] = {
    "small": "8-channel wearable IMU (orientation + acceleration + angular velocity) at 8 body sites, "
             "expressed in the spec sensor frame S (spec_S_v2).",
    "large": "Reduced-model kinematics: 18-joint rotations/velocities + pelvis trajectory, plus the "
             "lossless 24-joint SMPL global orientation.",
    "anthro": "Subject/sequence skeleton constants: rest joint positions, segment lengths, SMPL shape "
              "(betas), and body attributes.",
    "root_translation": "Raw SMPL root (pelvis) world-translation stream for the take.",
    "development_reference": "PRISM insole development reference (GRF / CoP / contacts); development-only, "
                             "never a generator input, supervision, or label.",
}

FIELD_DOCS: dict[str, dict[str, str]] = {
    "small": {
        "imu_orientation": "(T,8,4) unit quaternion wxyz — each site's sensor orientation in world (spec frame S); NaN where imu_valid_mask is False.",
        "imu_acceleration": "(T,8,3) m/s^2 specific force in the sensor frame, gravity-included (~9.8 at rest); NaN where imu_valid_mask is False.",
        "imu_angular_velocity": "(T,8,3) deg/s angular velocity in the sensor frame; NaN where imu_valid_mask is False.",
        "imu_valid_mask": "(T,8) bool per-frame per-site validity (False = gap / invalid). Where False the three IMU arrays are NaN in that cell and imu_confidence is 0.0; where True they are finite. Apply it before any filter, derivative or loss; large stays finite on every frame.",
        "imu_confidence": "(T,8) confidence weight per site; 0.0 where imu_valid_mask is False. The non-zero levels and what they mean are set per corpus and are NOT comparable between corpora — read the bundle's own manifest and description, never another bundle's.",
        "imu_orientation_absolute_heading": "(8,) bool — True if heading is absolute; False = gravity-referenced only (insole feet).",
        "sensor_codes": "(8,) site names in canonical order: back_T4, wrist_l, wrist_r, shank_l, shank_r, occiput, foot_l, foot_r.",
        "mount_id": "(8,) mount identifier per site (source::site).",
        "small_mode": "() how small was built: measured_physical (PRISM), synthetic_from_smpl (AMASS, AddBiomechanics, HKNU) or synthetic_from_markers (GAITEX).",
        "axis_convention": "() sensor-frame convention tag; always spec_S_v2.",
        "pair_id": "() stable take-identity hash (identical across small/large/anthro/root_translation).",
        "timestamps_s": "(T,) float64 seconds on the 100 Hz grid, starting at 0.",
        "frame_count": "() int64 number of frames T.",
        "q_anatomical_from_sensor": "(8,4) quat wxyz — constant per-site calibration A<-S_phys (measured only).",
        "p_segment_to_sensor_m": "(8,3) m sensor lever arm on the segment (zeros = at segment origin; measured only).",
        "imu_synth_mask": "(T,8) bool — physical frames that were gap-filled/synthetic (e.g. R_wrist ~56%; measured only).",
        "gyro_provenance": "(8,) how each site's gyro was obtained (measured gyr_local_raw vs derived from orientation; measured only).",
    },
    "large": {
        "joint_names": "(18,) reduced-model joint order (index 0 = pelvis global, 1-17 = SMPL local joints).",
        "joint_rotation": "(T,18,4) quat wxyz — [0] pelvis global orientation, [1:] 17 parent-relative local rotations.",
        "joint_velocity": "(T,18,3) deg/s angular velocity per joint ([0] = pelvis).",
        "root_velocity": "(T,3) m/s pelvis root velocity (frame-to-frame displacement / dt).",
        "pelvis_position_world_aux": "(T,3) m absolute pelvis path (anchor for integrating root_velocity).",
        "smpl_global_orientation_world": "(T,24,4) quat wxyz — lossless global orientation of all 24 SMPL joints in world.",
        "pair_id": "() stable take-identity hash (matches small/anthro).",
        "timestamps_s": "(T,) float64 seconds, 100 Hz grid.",
        "frame_count": "() int64 number of frames T.",
    },
    "anthro": {
        "namespace": "() anthro namespace tag.",
        "subject_scope": "() scope of these constants (subject-constant for PRISM, sequence-constant for AMASS).",
        "subject_scope_note": "() human note on the frame-invariance of the constants.",
        "reference_poses": "(2,) the two reference poses (e.g. T-pose, upright-neutral) that joint_position is given for.",
        "joint_names": "(22,) SMPL 22-chain joint names for joint_position.",
        "joint_position": "(2,22,3) m rest joint positions for the 2 reference poses (SMPL canonical rest, Y-up).",
        "joint_position_frame": "() frame of joint_position (smpl_canonical_rest_yup).",
        "segment_names": "(13,) the 13 body-segment names for segment_length.",
        "segment_length": "(13,) m bone/segment lengths.",
        "fixed_joint_names": "(4,) the 4 frozen joints of the reduced model (spine1, spine2, left/right collar).",
        "fixed_joint_rotation": "(4,4) quat wxyz — constant local rotation of each frozen joint, fitted once per subject by smpl18.reduce (reduced_model_fit.json at the bundle root).",
        "sex": "() subject sex.",
        "height": "() m stature.",
        "body_mass": "() kg body mass (NaN if unavailable, e.g. AMASS).",
        "betas": "(N,) SMPL shape coefficients; N = betas_num (10 for PRISM's model, 16 for AMASS).",
        "betas_num": "() int64 number of betas.",
        "gender": "() SMPL model gender used.",
        "height_provenance": "() provenance of height (measured / source_derived / estimated / undeclared).",
        "body_mass_provenance": "() provenance of body_mass.",
        "sex_provenance": "() provenance of sex.",
        "betas_provenance": "() provenance of betas.",
        "skeleton_provenance": "() provenance of the rest skeleton (joint_position / segment_length).",
        "fixed_joint_rotation_provenance": "() how the frozen-joint constants were fitted (method, scope, pooled takes and frames).",
        "reconstruction_note": "() how to reconstruct world joints from anthro rest + large joint_rotation.",
        "pair_id": "() stable take-identity hash (matches small/large).",
        "subject_id": "() source subject identifier.",
        "source_asset_id": "() SOMA_DATA_ROOT-relative id of the source asset this take came from.",
    },
    "root_translation": {
        "smpl_trans": "(T,3) m raw SMPL root (pelvis) translation from the source SMPL params, full take.",
        "frame_count": "() int64 number of frames T.",
        "pair_id": "() stable take-identity hash (matches the other artifacts).",
        "source_asset_id": "() SOMA_DATA_ROOT-relative id of the source asset.",
        "smpl_trans_provenance": "() provenance of smpl_trans (source_derived from the source SMPL params).",
    },
    "development_reference": {
        "namespace": "() dev-ref namespace (development_reference/prism_insole_v1).",
        "grf_feet_source_native": "(T,2,3) N per-foot ground reaction force in source-native world (order L, R).",
        "grf_combined_source_native": "(T,3) N combined ground reaction force.",
        "cop_feet_world": "(T,2,3) m per-foot center of pressure in world (L, R).",
        "cop_combined_world": "(T,3) m combined center of pressure in world.",
        "foot_contact_mask": "(T,2) bool per-foot contact state (L, R).",
        "vertical_force_provenance": "() provenance of the vertical force (measured).",
        "cop_provenance": "() provenance of CoP (source_derived).",
        "contacts_provenance": "() provenance of contacts (source_derived).",
        "usage": "() usage policy: development_reference_only (generator_input=false, supervision=false).",
        "pair_id": "() stable take-identity hash.",
        "timestamps_s": "(T,) float64 seconds, 100 Hz grid.",
    },
}

ENUM_DOCS: dict[str, dict[str, str]] = {
    "small_mode": {
        "measured_physical": "PRISM physical imu.* stream (SO(3)-projected) calibrated into the spec sensor frame S.",
        "synthetic_from_smpl": "AMASS: IMU synthesized from SMPL forward kinematics (no physical sensor).",
        "synthetic_from_markers": "GAITEX: IMU synthesized from optical marker clusters (rigid-body pose, plate lever arm, zero-phase low-pass), no physical sensor; orientation is not a constant relabel of the SMPL joint in large, and imu_valid_mask carries per-site marker gaps (NaN under False).",
    },
    "axis_convention": {
        "spec_S_v2": "+Y=proximal, +Z=outward-lateral, +X=Y x Z; left limbs mounted 180 deg about Y (left +X=posterior).",
    },
    "subject_scope": {
        "subject_constant_trial_frame_invariant": "constants fixed per subject, invariant across all frames of a take.",
        "sequence_constant_frame_invariant": "constants fixed per sequence, invariant across all frames.",
    },
}


def summary(artifact: str, field: str) -> str:
    """One-line semantics for a field. Raises KeyError for unknown artifact/field."""
    return FIELD_DOCS[artifact][field]
