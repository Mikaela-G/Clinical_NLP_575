#!/bin/sh
python3 data_loader.py \
    /home2/mguerr20/i2b2_data/concept_assertion_relation_training_data/beth/txt \
    /home2/mguerr20/i2b2_data/concept_assertion_relation_training_data/beth/concept \
    /home2/mguerr20/i2b2_data/concept_assertion_relation_training_data/partners/txt \
    /home2/mguerr20/i2b2_data/concept_assertion_relation_training_data/partners/concept \
    /home2/mguerr20/i2b2_data/test_data \
    /home2/mguerr20/i2b2_data/reference_standard_for_test_data
    # /home2/mguerr20/i2b2_data/small_examples/txt \
    # /home2/mguerr20/i2b2_data/small_examples/concept \