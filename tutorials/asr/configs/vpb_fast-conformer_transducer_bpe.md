
```Mermaid
flowchart TB
    subgraph Input
        Audio[Audio Input
        16kHz]
    end

    subgraph Preprocessor
        MEL["MEL Spectrogram 
        80 features 
        Window size: 25ms 
        Stride: 10ms"]
        SpecAug[SpecAugment
        2 freq masks
        10 time masks]
    end

    subgraph Encoder["Encoder (Fast Conformer)"]
        direction TB
        Sub[Subsampling Layer
        DW Striding x8]
        ConformerBlocks[16 Conformer Blocks
        d_model=176]
        subgraph ConformerBlock["Single Conformer Block"]
            direction TB
            FF1[Feed Forward Module
	            Expansion Factor: 4]
            MHSA[Multi-Head Self Attention
	            8 heads]
            Conv[Convolution Module
                Kernel Size: 9]
            FF2[Feed Forward Module]
        end
    end

    subgraph Decoder["Decoder (RNN-T)"]
        direction TB
        PredNet[Prediction Network	
                1 LSTM Layer	
                Hidden: 640]
    end

    subgraph Joint["Joint Network"]
        direction TB
        JointHidden[Joint Hidden Layer	
                    Size: 640
	                Activation: ReLU]
        Output[Output Layer
	            Softmax]
    end

    Audio --> MEL
    MEL --> SpecAug
    SpecAug --> Sub
    Sub --> ConformerBlocks
    ConformerBlocks --> |Encoder Output| JointHidden
    PredNet --> |Decoder Output| JointHidden
    JointHidden --> Output

    style Encoder fill:#f9f,stroke:#333
    style Decoder fill:#bbf,stroke:#333
    style Joint fill:#bfb,stroke:#333
    style Preprocessor fill:#fbb,stroke:#333

```
