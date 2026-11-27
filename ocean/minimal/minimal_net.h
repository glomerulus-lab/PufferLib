// CPU minimal policy: entity encoder + RNN + decoder.
// Weight order matches CUDA weights_create / reg_params:
//   encoder input_w (16 x 6), encoder output_w (hidden x 16),
//   decoder ((9+5+1) x hidden), recurrent layers (per [policy] network; see RNN in puffercpu.c).
// Forward matches ocean/minimal/minimal.cu:
//   materialize 16 entities [self(2) | point_i(4)], Linear 6->16, ReLU,
//   Linear 16->hidden, max-pool over the 16 points, then RNN + decoder.
// Include puffercpu.c first (Linear / ReLU / RNN).

#define ME_SELF_DIM 2
#define ME_POINT_DIM 4
#define ME_NUM_POINTS 16
#define ME_ENTITY_IN (ME_SELF_DIM + ME_POINT_DIM)
#define ME_ENTITY_HIDDEN 16
#define ME_OBS_SIZE (ME_SELF_DIM + ME_NUM_POINTS * ME_POINT_DIM)
#define ME_NUM_ATNS 2

#if defined(OBS_SIZE) && (OBS_SIZE != ME_OBS_SIZE)
#error "minimal entity encoder expects OBS_SIZE 66"
#endif

typedef struct MinimalNet MinimalNet;
struct MinimalNet {
    int num_agents;
    int hidden;
    float* point_input;
    float* encoded;
    Linear* entity_fc;
    ReLU* entity_relu;
    Linear* out_fc;
    Linear* decoder;
    RNN* rnn;
    Multidiscrete* multidiscrete;
};

static inline int puf_me_align8(int n) {
    return (n + 7) & ~7;
}

static inline int minimal_weight_count(NetworkType network, int hidden, int layers) {
    int n = 0;
    n = puf_me_align8(n + ME_ENTITY_HIDDEN * ME_ENTITY_IN);
    n = puf_me_align8(n + hidden * ME_ENTITY_HIDDEN);
    n = puf_me_align8(n + (9 + 5 + 1) * hidden);
    return rnn_weight_count(network, n, hidden, layers);
}

static inline MinimalNet* init_minimal_net(Weights* weights, NetworkType network, int num_agents,
        int hidden, int layers) {
    MinimalNet* net = (MinimalNet*)calloc(1, sizeof(MinimalNet));
    net->num_agents = num_agents;
    net->hidden = hidden;
    net->point_input = (float*)calloc((size_t)num_agents * ME_NUM_POINTS * ME_ENTITY_IN,
        sizeof(float));
    net->encoded = (float*)calloc((size_t)num_agents * hidden, sizeof(float));
    net->entity_fc = make_linear(weights, num_agents * ME_NUM_POINTS,
        ME_ENTITY_IN, ME_ENTITY_HIDDEN);
    net->entity_relu = make_relu(num_agents * ME_NUM_POINTS, ME_ENTITY_HIDDEN);
    net->out_fc = make_linear(weights, num_agents * ME_NUM_POINTS,
        ME_ENTITY_HIDDEN, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, 9 + 5 + 1);
    net->rnn = make_rnn(weights, network, num_agents, hidden, layers);
    int logit_sizes[ME_NUM_ATNS] = {9, 5};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, ME_NUM_ATNS);
    return net;
}

static inline void free_minimal_net(MinimalNet* net) {
    free(net->point_input);
    free(net->encoded);
    free(net->entity_fc);
    free(net->entity_relu);
    free(net->out_fc);
    free(net->decoder);
    free_rnn(net->rnn);
    free(net->multidiscrete);
    free(net);
}

static inline void minimal_encode(MinimalNet* net, const float* observations) {
    int B = net->num_agents;
    int H = net->hidden;
    for (int b = 0; b < B; b++) {
        const float* obs = observations + b * ME_OBS_SIZE;
        for (int p = 0; p < ME_NUM_POINTS; p++) {
            float* ent = net->point_input
                + ((size_t)b * ME_NUM_POINTS + p) * ME_ENTITY_IN;
            memcpy(ent, obs, ME_SELF_DIM * sizeof(float));
            memcpy(ent + ME_SELF_DIM,
                obs + ME_SELF_DIM + p * ME_POINT_DIM,
                ME_POINT_DIM * sizeof(float));
        }
    }
    linear(net->entity_fc, net->point_input);
    relu(net->entity_relu, net->entity_fc->output);
    linear(net->out_fc, net->entity_relu->output);
    _max_dim1(net->out_fc->output, net->encoded, B, ME_NUM_POINTS, H);
}

static inline void forward_minimal(MinimalNet* net, const float* observations,
        float* terminals, float* actions) {
    rnn_zero_term(net->rnn, terminals);
    minimal_encode(net, observations);
    rnn_forward(net->rnn, net->encoded);
    linear(net->decoder, net->rnn->output);
    multidiscrete(net->multidiscrete, net->decoder->output, actions, 0, NULL);
}
