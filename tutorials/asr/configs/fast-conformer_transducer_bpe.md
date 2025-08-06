``` Mermaid
graph TD
    subgraph Model Architecture
        direction LR
        A[preprocessor] --> B(encoder) --> C(decoder) --> D(joint)
        subgraph Augmentation
            E[spec_augmentation]
        end
    end

    subgraph "Preprocessor Details"
        A[preprocessor<br/>AudioToMelSpectrogramPreprocessor] --> A1[featurizer<br/>FilterbankFeatures]
    end

    subgraph "Encoder Details"
        B(encoder<br/>ConformerEncoder<br/>5.1 M) --> B1(pre_encode<br/>ConvSubsampling<br/>590 K)
        B --> B2(pos_enc<br/>RelPositionalEncoding)
        B --> B3(layers<br/>ModuleList<br/>4.5 M)
    end

    subgraph "Encoder Pre-encoding"
        B1 --> B1a(out<br/>Linear<br/>450 K)
        B1 --> B1b(conv<br/>Sequential<br/>139 K)
        B1b --> B1b1("Conv2d, ReLU, Conv2d, ...")
    end

    subgraph "Encoder Positional Encoding"
        B2 --> B2a(dropout<br/>Dropout)
    end

    subgraph "Encoder Layers"
        B3 --> B3_0("layer 0<br/>ConformerLayer<br/>750 K")
        B3 --> B3_1("layer 1<br/>ConformerLayer<br/>750 K")
        B3 --> B3_etc(...)
        B3 --> B3_5("layer 5<br/>ConformerLayer<br/>750 K")
    end

    subgraph "Conformer Layer 0 Breakdown"
        B3_0 --> L0_ff1(feed_forward1<br/>ConformerFeedForward<br/>248 K)
        B3_0 --> L0_conv(conv<br/>ConformerConvolution<br/>95.6 K)
        B3_0 --> L0_attn(self_attn<br/>RelPositionMultiHeadAttention<br/>155 K)
        B3_0 --> L0_ff2(feed_forward2<br/>ConformerFeedForward<br/>248 K)
    end

    subgraph "Feed Forward Detail (L0_ff1)"
        L0_ff1 --> L0_ff1_l1(linear1<br/>Linear<br/>124 K)
        L0_ff1 --> L0_ff1_act(activation<br/>Swish)
        L0_ff1 --> L0_ff1_drop(dropout<br/>Dropout)
        L0_ff1 --> L0_ff1_l2(linear2<br/>Linear<br/>124 K)
    end

    subgraph "Convolution Detail (L0_conv)"
        L0_conv --> L0_conv_p1(pointwise_conv1<br/>Conv1d<br/>62.3 K)
        L0_conv --> L0_conv_d(depthwise_conv<br/>CausalConv1D<br/>1.8 K)
        L0_conv --> L0_conv_bn(batch_norm<br/>BatchNorm1d)
        L0_conv --> L0_conv_act(activation<br/>Swish)
        L0_conv --> L0_conv_p2(pointwise_conv2<br/>Conv1d<br/>31.2 K)
    end

    subgraph "Self Attention Detail (L0_attn)"
        L0_attn --> L0_attn_q(linear_q<br/>Linear)
        L0_attn --> L0_attn_k(linear_k<br/>Linear)
        L0_attn --> L0_attn_v(linear_v<br/>Linear)
        L0_attn --> L0_attn_out(linear_out<br/>Linear)
        L0_attn --> L0_attn_pos(linear_pos<br/>Linear)
    end

    subgraph "Decoder Details"
        C(decoder<br/>RNNTDecoder<br/>863 K) --> C1(prediction<br/>ModuleDict<br/>863 K)
        C1 --> C1a(embed<br/>Embedding<br/>41.3 K)
        C1 --> C1b(dec_rnn<br/>LSTMDropout<br/>821 K)
        C1b --> C1b1("lstm<br/>LSTM")
        C1b --> C1b2("dropout<br/>Dropout")
    end

    subgraph "Joint Details"
        D(joint<br/>RNNTJoint<br/>200 K) --> D1(pred<br/>Linear<br/>102 K)
        D --> D2(enc<br/>Linear<br/>56.6 K)
        D --> D3(joint_net<br/>Sequential<br/>41.4 K)
        D --> D4(_loss<br/>RNNTLoss)
        D --> D5(_wer<br/>WER)
        D3 --> D3a("ReLU, Dropout, Linear")
        D4 --> D4a("_loss<br/>RNNTLossNumba")
    end

    style A fill:#cde4ff,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#d5e8d4,stroke:#333,stroke-width:2px
    style C fill:#fff2cc,stroke:#333,stroke-width:2px
    style D fill:#f8cecc,stroke:#333,stroke-width:2px

```