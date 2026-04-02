from __future__ import annotations

import os

from test_types_basic import test_types_basic
from test_song_base import test_song_base
from test_mod_basic import (
    test_mod_note_helpers,
    test_mod_basic_ops,
    test_mod_row_count_and_duration,
    test_mod_effect_setters,
    test_mod_misc,
    test_mod_edge_cases,
    test_mod_multi_channel_patterns,
    test_mod_random_edit_stress,
    test_mod_channel_ops,
    test_mod_mute_channel_global_effects,
    test_mod_pattern_ops,
    test_mod_validation_helpers,
    test_mod_sample_helpers,
)
from test_mod_samples_roundtrip import (
    test_mod_sample_ops,
    test_mod_random_roundtrip,
    test_mod_sample_copy_roundtrip,
)
from test_xm_basic import (
    test_xm_basic_ops,
    test_xm_patterns_edge_cases,
    test_xm_sample_loop_edges,
    test_xm_random_edit_stress,
    test_xm_instrument_map_stress,
    test_pattern_sequence_stress,
    test_xm_misc,
    test_xm_channel_ops,
    test_xm_mute_channel_global_effects,
    test_xm_pattern_ops,
)
from test_xm_instruments import (
    test_xm_instrument_copy,
    test_xm_instrument_sample_ops,
    test_xm_instrument_helpers,
    test_xm_instrument_edge_cases,
    test_xm_validation_helpers,
    test_xm_sample_helpers,
)
from test_loaded_files import (
    test_loaded_mod_files,
    test_loaded_xm_files,
)
from test_s3m_basic import (
    test_s3m_basic_types,
    test_s3m_song_defaults,
    test_s3m_basic_editing,
    test_s3m_sequence_limits,
    test_s3m_channel_ops,
    test_s3m_mute_channel_preserves_global_effects,
    test_s3m_header_load_synthetic,
    test_s3m_header_load_real_files,
)
from test_s3m_sample_api import (
    test_s3m_load_pcm_sample_8bit,
    test_s3m_load_pcm_sample_16bit,
    test_s3m_rejects_adlib_instruments,
    test_s3m_load_real_pcm_samples,
)
from test_s3m_pattern_decode import (
    test_s3m_pattern_decode_synthetic,
    test_s3m_pattern_decode_real_files,
)
from test_s3m_roundtrip_write import (
    test_s3m_generated_roundtrip,
    test_s3m_corpus_semantic_roundtrip,
)
from test_s3m_timing_functions import (
    test_s3m_speed_and_tempo_timestamps,
    test_s3m_pattern_break_and_order_jump,
    test_s3m_pattern_delay_and_effective_rows,
)


def run_all() -> None:
    root = r"G:\My Drive\Moduli"
    count = int(os.environ.get("NODMOD_SAMPLE_COUNT", "10"))

    tmp_dir = os.path.join(os.getcwd(), "dev")

    test_types_basic()
    test_song_base()

    test_mod_note_helpers()
    test_mod_basic_ops()
    test_mod_row_count_and_duration()
    test_mod_sample_ops(tmp_dir)
    test_mod_effect_setters()
    test_mod_misc(tmp_dir)
    test_mod_edge_cases()
    test_mod_multi_channel_patterns()
    test_mod_random_edit_stress()
    test_mod_channel_ops()
    test_mod_mute_channel_global_effects()
    test_mod_pattern_ops()
    test_mod_validation_helpers()
    test_mod_sample_helpers()
    test_mod_random_roundtrip(tmp_dir)
    test_mod_sample_copy_roundtrip(root, tmp_dir)

    test_xm_basic_ops()
    test_xm_instrument_copy()
    test_xm_instrument_sample_ops(tmp_dir)
    test_xm_instrument_helpers(tmp_dir)
    test_xm_validation_helpers()
    test_xm_sample_helpers()
    test_xm_instrument_edge_cases()
    test_xm_patterns_edge_cases()
    test_xm_sample_loop_edges()
    test_xm_random_edit_stress()
    test_xm_instrument_map_stress()
    test_xm_channel_ops()
    test_xm_mute_channel_global_effects()
    test_xm_pattern_ops()
    test_pattern_sequence_stress()
    test_xm_misc(tmp_dir)

    test_s3m_basic_types()
    test_s3m_song_defaults()
    test_s3m_basic_editing()
    test_s3m_sequence_limits()
    test_s3m_channel_ops()
    test_s3m_mute_channel_preserves_global_effects()
    test_s3m_header_load_synthetic()
    test_s3m_header_load_real_files()
    test_s3m_load_pcm_sample_8bit()
    test_s3m_load_pcm_sample_16bit()
    test_s3m_rejects_adlib_instruments()
    test_s3m_load_real_pcm_samples()
    test_s3m_pattern_decode_synthetic()
    test_s3m_pattern_decode_real_files()
    test_s3m_generated_roundtrip()
    test_s3m_corpus_semantic_roundtrip()
    test_s3m_speed_and_tempo_timestamps()
    test_s3m_pattern_break_and_order_jump()
    test_s3m_pattern_delay_and_effective_rows()

    test_loaded_mod_files(root, count, tmp_dir)
    test_loaded_xm_files(root, count, tmp_dir)

    print("OK: full suite passed")


if __name__ == "__main__":
    run_all()
