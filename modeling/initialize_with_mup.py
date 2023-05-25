from modeling.mup_utils import get_lazy_model_from_scratch
import math
import copy

def mup_init_from_scratch(config, training_args, model_args, logger):
    logger.info(f'Loading Mup model from scratch')
    mup = training_args.use_mup
    size_per_head = training_args.size_per_head

    config.output_mult = training_args.output_mult
    config.initializer_range = training_args.initializer_range

    # maybe reset dropout to zero
    if training_args.unified_dropout is not None:
        print(f"resetting dropout={training_args.unified_dropout}")
        config.attn_pdrop = training_args.unified_dropout
        config.embd_pdrop = training_args.unified_dropout
        config.resid_pdrop = training_args.unified_dropout
        config.summary_first_dropout = training_args.unified_dropout

    config_base = copy.deepcopy(config)

    logger.info(f"Generating proxy model for HP tuning")
    config_hp_search = copy.deepcopy(config_base)
    config_hp_search.attn_mult = float(math.sqrt(size_per_head)) if mup else None
    config_hp_search.n_embd = training_args.hp_tune_actual_width
    config_hp_search.n_head = int(config_hp_search.n_embd / training_args.size_per_head)

    model_f = get_lazy_model_from_scratch(config=config_hp_search,
                            mup=mup,
                            readout_zero_init=mup and training_args.readout_zero_init,
                            query_zero_init=mup and training_args.query_zero_init,
                            input_mult=training_args.output_mult,
                            width_mult_for_weights=training_args.width_mult_for_weights)
    model = model_f()
    model.transformer.input_mult = training_args.output_mult
    assert model.transformer.input_mult is not None

    return model