---
layout: default
title: Diagrams
permalink: /diagrams.html
---

# Architecture Diagrams

---

## 1. System Architecture

High-level components and their responsibilities.

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        AP["ArgParse.c<br/>parse argv → CommandArgs"]
        CD["Commands.c<br/>dispatch by CMD_*"]
        TR["Train.c"]
        TE["Test.c"]
        BM["Benchmark.c"]
        RC["Recognize.c"]
    end

    subgraph CORE["Core Layer"]
        GL["Glue.c<br/>forward pass - loss - weight update"]
        GR["Grad.c<br/>ReLU derivative"]
        MO["Model.c<br/>create - save - load .bin"]
        TE2["Tensor.c<br/>dot - add - ReLU - softmax - Xavier"]
        AR["Arena.c<br/>slab allocator"]
    end

    subgraph DATA["Dataset Layer"]
        DS["Dataset.c<br/>MEMORY or PNG"]
        TU["TestUtils.c<br/>trainModel - testPerfect - robustness - confusion"]
        MD["MemoryDatasets.c<br/>hardcoded float arrays"]
    end

    subgraph IMAGE["Image Layer"]
        IL["ImageLoader.c<br/>stb_image PNG -> RawImage"]
        IP["ImagePreprocess.c<br/>grayscale - Otsu - resize - normalize"]
        SG["Segmenter.c<br/>vertical projection - <br> char bounds - space detection"]
    end

    AP --> CD
    CD --> TR & TE & BM & RC
    TR & TE & BM --> DS & TU
    RC --> DS
    TR & TE & BM & RC --> MO
    TU --> GL
    GL --> GR & TE2 & AR
    MO --> TE2 & AR
    DS --> MD
    DS --> IP & IL
    RC --> SG
    SG --> IP & IL

    style CLI fill:#dbeafe,stroke:#3b82f6
    style CORE fill:#dcfce7,stroke:#22c55e
    style DATA fill:#fef9c3,stroke:#eab308
    style IMAGE fill:#fce7f3,stroke:#ec4899
```

---

## 2. Module Dependencies

Actual `#include` relationships between `.c` files and headers.

```mermaid
flowchart LR

%% =====================
%% IMAGE PIPELINE (Lowest Level)
%% =====================
subgraph IMAGE["Image Processing Pipeline"]
    direction TB
    hIL["ImageLoader.h"]
    hIP["ImagePreprocess.h"]
    hSG["Segmenter.h"]

    Loader["ImageLoader.c"]
    Preprocess["ImagePreprocess.c"]
    Segmenter["Segmenter.c"]
end

%% =====================
%% DATASET / TEST
%% =====================
subgraph DATASET["Dataset & Testing"]
    direction TB
    hDS["Dataset.h"]
    hTU["TestUtils.h"]

    Dataset["Dataset.c"]
    TestUtils["TestUtils.c"]
end

%% =====================
%% CORE ML ENGINE
%% =====================
subgraph CORE["Core ML Engine"]
    direction TB
    hAR["Arena.h"]
    hTE["Tensor.h"]
    hMO["Model.h"]
    hGR["Grad.h"]
    hGL["Glue.h"]

    Arena["Arena.c"]
    Tensor["Tensor.c"]
    Model["Model.c"]
    Grad["Grad.c"]
    Glue["Glue.c"]
end

%% =====================
%% CLI LAYER
%% =====================
subgraph CLI["CLI Layer"]
    direction TB
    hAP["ArgParse.h"]
    hCD["Commands.h"]

    subgraph COMMANDS["cli/commands"]
        direction TB
        cTRAIN["Train.c"]
        cTEST["Test.c"]
        cBENCH["Bench.c"]
        cREC["Rec.c"]
    end
end

%% =====================
%% ENTRY POINT (Top Level)
%% =====================
subgraph ENTRY["Entry Point"]
    main["miniAI.c"]
end

%% =====================
%% GLOBAL CONFIG
%% =====================
hAI["AIHeader.h<br/>(TrainingConfig / Constants)"]

%% =====================
%% VERTICAL ORDER CONTROL (Invisible)
%% =====================
IMAGE -.-> DATASET
DATASET -.-> CORE
CORE -.-> CLI
CLI -.-> ENTRY

linkStyle 0 stroke-width:0px
linkStyle 1 stroke-width:0px
linkStyle 2 stroke-width:0px
linkStyle 3 stroke-width:0px

%% =====================
%% ENTRY DEPENDENCIES
%% =====================
main --> hAP
main --> hCD
main --> hAI
main --> cTRAIN

%% =====================
%% CLI DEPENDENCIES
%% =====================
COMMANDS --> hCD
COMMANDS --> hDS
COMMANDS --> hAI

cTRAIN --> hTU
cTRAIN --> hMO

cTEST --> hTU
cTEST --> hMO
cTEST --> hGL

cBENCH --> hTU
cBENCH --> hMO

cREC --> hMO
cREC --> hGL

%% =====================
%% CORE IMPLEMENTATION
%% =====================
Glue --> hGL
Glue --> hGR
Glue --> hAI

Grad --> hGR
Model --> hMO
Tensor --> hTE
Arena --> hAR

%% =====================
%% DATASET IMPLEMENTATION
%% =====================
Dataset --> hDS
Dataset --> hIL
Dataset --> hIP

TestUtils --> hTU
TestUtils --> hGL
TestUtils --> hAI

%% =====================
%% IMAGE IMPLEMENTATION
%% =====================
Segmenter --> hSG
Segmenter --> hIP

Loader --> hIL
Preprocess --> hIP

%% =====================
%% HEADER RELATIONSHIPS
%% =====================
hGL --> hTE
hGL --> hAR

hMO --> hTE
hMO --> hAR

hTU --> hMO
hTU --> hDS

hDS --> hIL
hDS --> hIP

hSG --> hIP

%% =====================
%% STYLING
%% =====================
style hAI fill:#fef3c7,stroke:#f59e0b
```

---

## 3. Training Data Flow

Exact sequence executed by `cmdTrain()` → `trainModel()`.

```mermaid
flowchart TB
    A(["`./miniAI train --dataset digits --data`"]) --> B["parseArgs()<br/>→ CommandArgs"]
    B --> C{"useStatic?"}

    C -- yes --> D["datasetCreateMemory()<br/>float* from MemoryDatasets.c<br/>digits: 5×5, alpha: 8×8"]
    C -- no  --> E["datasetCreatePNG()<br/>load PNGs at startup<br/>digits: 8×8, alpha: 16×16"]

    D & E --> F["loadBestConfig()<br/>IO/configs/best_config_*.txt<br/>→ g_trainConfig.hiddenSize/LR"]

    F --> G["modelCreate(perm)<br/>Xavier init weights<br/>bias = 0"]

    G --> H["trainModel() — 3000 passes"]

    subgraph loop["Each pass (epoch)"]
        H --> I["shuffle(indices)"]
        I --> J["for each sample"]
        J --> K["arenaReset(scratch)"]
        K --> L["glueAccumulateGradients()<br/>forward + backward<br/>clip gradients, accumulate"]
        L --> M{"batch full?<br/>or last sample?"}
        M -- yes --> N["glueUpdateWeights()<br/>avg grad ÷ batchSize<br/>+ L2 regularization<br/>reset accumulators"]
        N --> J
        M -- no --> J
        J --> O{"pass % 500 == 0?"}
        O -- yes --> P["lr *= 0.7<br/>log loss"]
        P --> H
    end

    H --> Q["modelSave()<br/>IO/models/*.bin"]
    Q --> R["testPerfect() · testRobustness()<br/>visualDemo() · confusionMatrix()"]
    R --> S["arenaFree(perm, scratch)"]

    style loop fill:#f0fdf4,stroke:#22c55e
```

---

## 4. Inference Data Flow

Sequence for `cmdTest()` with a single image (`--image`).

```mermaid
flowchart TD
    A(["`./miniAI test --image char.png`"]) --> B["parseArgs()<br/>useStatic=0, PNG mode"]
    B --> C["loadBestConfig()<br/>hidden size from config"]
    C --> D["modelCreate(perm)<br/>empty shell, correct dims"]
    D --> E["modelLoad()<br/>verify layer dims<br/>read weights + biases"]

    E --> F["imageLoad() via stb_image<br/>→ RawImage width×height×channels"]
    F --> G["convertToGrayscale()<br/>luminance: 0.299R + 0.587G + 0.114B"]
    G --> H["calculateOtsuThreshold()<br/>maximize between-class variance"]
    H --> I["binarize + resize to gridSize×gridSize<br/>normalize → float[0.0, 1.0]"]

    I --> J["gluePredict(model, input, scratch)<br/>records scratch->used"]

    subgraph fwd["glueForward()"]
        J --> K["for each layer i:"]
        K --> L["z = W·input + b<br/>tensorDot + tensorAdd"]
        L --> M{"last layer?"}
        M -- no  --> N["a = ReLU(z)"]
        M -- yes --> O["a = z  (raw logits)"]
        N & O --> K
    end

    O --> P["tensorSoftmax(probs, output)<br/>numerical stability: subtract max"]
    P --> Q["argmax(probs) → predicted class"]
    Q --> R["scratch->used = startPos<br/>(implicit free of temporaries)"]
    R --> S[/"charMap[guess] + confidence%"/]

    style fwd fill:#eff6ff,stroke:#3b82f6
```

---

## 5. Phrase Recognition Flow

How `cmdRecognize()` processes a full phrase image.

```mermaid
flowchart TD
    A(["`./miniAI recognize --image phrase.png`"]) --> B["datasetCreatePhrase()<br/>calls segmentPhrase()"]

    subgraph seg["Segmenter.c — segmentPhrase()"]
        B --> C["imageLoad() → RawImage"]
        C --> D["convertToGrayscale()"]
        D --> E["shouldInvertColors()<br/>sample border pixels<br/>avgBorder > 128 → invert"]
        E --> F["calculateOtsuThreshold()"]
        F --> G["binarize → uint8_t binary[]"]
        G --> H["computeVerticalProjection()<br/>sum foreground pixels per column"]
        H --> I["findCharBoundaries()<br/>detect runs of non-zero columns<br/>record gap widths between chars"]
        I --> J["detectSpaces()<br/>gaps > spaceThreshold → insert ' '"]
        J --> K["for each character segment:<br/>crop → resize → normalize<br/>→ float[gridSize²]"]
    end

    K --> L["Dataset with N character samples<br/>+ space flags"]

    L --> M["loadBestConfig() auto-detect<br/>from model filename"]
    M --> N["modelCreate() + modelLoad()"]

    N --> O["for each character sample"]
    O --> P["gluePredict() → class index"]
    P --> Q["charMap[index] → character"]
    Q --> R{"space flag?"}
    R -- yes --> S["append ' '"]
    R -- no  --> T["append char"]
    S & T --> O

    O --> U[/"Print assembled phrase<br/>+ per-character confidence"/]

    style seg fill:#fdf4ff,stroke:#a855f7
```

---

## 6. Memory Architecture

How the two arenas are used during training.

```mermaid
block-beta
    columns 3

    PERM_TITLE["<b>PERMANENT ARENA</b><br/>16 MB (Train) / 8 MB (Rec)"]:3
    block:perm:3
        columns 3
        P1["Model struct"]
        P2["Layer[0]:<br/>W, b, gradW, gradB"]
        P3["Layer[1]:<br/>W, b, gradW, gradB"]
        P_INFO["<i>Never reset during training lifecycle</i>"]:3
    end

    space:3

    SCRATCH_TITLE["<b>SCRATCH ARENA</b><br/>4 MB (Train) / 2 MB (Rec)"]:3
    block:scratch:3
        columns 3
        S1["Input Tensor<br/>(One Sample)"]
        S2["Forward Cache:<br/>z, a (L0)"]
        S3["Forward Cache:<br/>z, a (L1)"]
        S4["Probs Tensor<br/>(Softmax)"]
        S5["Delta / Upstream<br/>(Backprop)"]
        S_INFO["<i>arenaReset() called per sample [O(1)]</i>"]:3
    end

    space:3

    NOTE_TITLE["<b>Memory Implementation Details</b>"]:3
    N1["<b>arenaAlloc:</b><br/>Pointer bump + memset(0)"]:1
    N2["<b>arenaReset:</b><br/>arenaReset = used = 0 (O(1))"]:1
    N3["<b>arenaFree:</b><br/>free(buffer) + free(struct)"]:1

    style PERM_TITLE fill:#dcfce7,stroke:#16a34a
    style SCRATCH_TITLE fill:#fee2e2,stroke:#dc2626
    style NOTE_TITLE fill:#f1f5f9,stroke:#475569
    style P_INFO fill:none,stroke:none
    style S_INFO fill:none,stroke:none
```

---

## 7. Neural Network — Forward Pass

Exact operations in `glueForward()` for a 2-layer network (input → hidden → output).

```mermaid
flowchart TB
    IN["input<br>float[inputSize x 1]"]

    subgraph L0["Layer 0 — hidden"]
        W0["W₀ <br>  [hidden x input]"]
        Z0["z₀ = W₀·input + b₀<br>tensorDot + tensorAdd<br>OpenMP parallel"]
        A0["a₀ = ReLU(z₀)<br>max(0, x)"]
    end

    subgraph L1["Layer 1 — output"]
        W1["W₁<br>[output x hidden]"]
        Z1["z₁ = W₁·a₀ + b₁<br>tensorDot + tensorAdd"]
        A1["a₁ = z₁<br>(raw logits)"]
    end

    subgraph POST["Post-forward"]
        SM["softmax(a₁)<br>subtract max for stability<br> -> probabilities"]
        OUT["argmax -> predicted class<br>probs[guess] -> confidence"]
    end

    IN --> W0 --> Z0 --> A0
    A0 --> W1 --> Z1 --> A1
    A1 --> SM --> OUT

    style L0 fill:#dbeafe,stroke:#3b82f6
    style L1 fill:#dcfce7,stroke:#22c55e
    style POST fill:#fef9c3,stroke:#eab308
```

---

## 8. Backpropagation — One Sample

Exact algorithm in `glueAccumulateGradients()`.

```mermaid
flowchart TD
    A["rawData[inputSize]"] --> B["apply TRAIN_NOISE=10%<br>flip pixel: val = 1 - val"]
    B --> C["glueForward()<br>cache z and a for all layers"]
    C --> D["tensorSoftmax(output) -> probs"]

    D --> E["output delta:<br>delta[i] = probs[i] - target[i]<br>target = one-hot(label)"]

    subgraph back["Backward loop layer i"]
        E --> F["for each weight r,c:<br>gradW = delta[r] x prevA[c]<br>clip to ±GRAD_CLIP<br>gradW_acc[r,c] += gradW"]
        F --> G["gradB_acc[r] += delta[r]"]
        G --> H{"i > 0 ?"}
        H -- yes --> I["upstreamDelta[j] = Σ W[k,j] x delta[k]<br>(W^T · delta)"]
        I --> J["tensorReLUDerivative()<br>nextDelta[i] = z[i]>0 ? upstream[i] : 0<br>OpenMP parallel"]
        J --> K["delta = nextDelta<br>-> next iteration"]
        K --> F
        H -- no  --> L["done accumulating"]
    end

    subgraph update["glueUpdateWeights() - called after full batch"]
        L --> M["avg = gradW_acc / batchSize"]
        M --> N["grad = avg + λ x W<br>(L2 regularization λ=0.0001)"]
        N --> O["W -= lr x grad<br>b -= lr x gradB_avg<br>reset accumulators to 0"]
    end

    style back fill:#fdf4ff,stroke:#a855f7
    style update fill:#f0fdf4,stroke:#22c55e
```

---

## 9. Dataset Type Comparison

```mermaid
flowchart LR
    subgraph static_ds["DATASET_MEMORY (--static)"]
        direction TB
        SM1["MemoryDatasets.c<br>hardcoded float arrays"]
        SM2["digits: 10 samples x 25 floats<br>(5×5 grid)"]
        SM3["alpha: 62 samples x 64 floats<br>(8×8 grid)"]
        SM4["datasetGetSample(ds, i)<br>= ds->memory.data + ixinputSize"]
        SM1 --> SM2 & SM3 --> SM4
    end

    subgraph png_ds["DATASET_PNG (--data)"]
        direction TB
        PM1["PNGs loaded at datasetCreatePNG()"]
        PM2["digits: 10 PNGs -> float[64]<br>(8×8 grid)"]
        PM3["alpha: 62 PNGs -> float[256]<br>(16×16 grid)"]
        PM4["datasetGetSample(ds, i)<br> = ds->png.samples[i]"]
        PM5["filename pattern:<br>grid==16 -> 065_A.png<br>grid==8 -> A.png"]
        PM1 --> PM2 & PM3 --> PM4
        PM1 --> PM5
    end

    subgraph phrase_ds["DATASET_PHRASE (recognize)"]
        direction TB
        PH1["datasetCreatePhrase(imageFile)"]
        PH2["segmentPhrase() -> CharSequence"]
        PH3["N char segments -> float[gridSize^2] each"]
        PH4["space flags array"]
        PH1 --> PH2 --> PH3 & PH4
    end

    style static_ds fill:#fef9c3,stroke:#eab308
    style png_ds fill:#dbeafe,stroke:#3b82f6
    style phrase_ds fill:#fce7f3,stroke:#ec4899
```