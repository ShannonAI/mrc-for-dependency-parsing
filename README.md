# MRC for Dependency Parsing

## Introduction
This repo contains code for paper [Dependency Parsing as MRC-based Span-Span Prediction](todo link)

## Results
Table 1: Results for different model on PTB and CTB.
<table border=2>
   <tr>
      <td></td>
      <td align='center' colspan="2">PTB</td>
      <td align='center' colspan="2">CTB</td> 
   </tr>
   <tr>
      <td></td>
      <td>UAS</td>
      <td>LAS</td>
      <td>UAS</td>
      <td>LAS</td>
   </tr>
   <tr>
      <td>StackPTR</td>
      <td>95.87</td>
      <td>94.19</td>
      <td>90.59</td>
      <td>89.29</td>
   </tr>
   <tr>
      <td>GNN</td>
      <td>95.87</td>
      <td>94.15</td>
      <td>90.78</td>
      <td>89.50</td>
   </tr>
   <tr>
      <td align='center' colspan="5">    +Pretrained Models</td>
   </tr>
   <tr>
      <td align="center" colspan="5"> with additional labelled constituency parsing data</td>
   </tr>
   <tr>
      <td>HPSG&#x266d</td>
      <td>97.20</td>
      <td>95.72</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>HPSG+LA&#x266d</td>
      <td>97.42</td>
      <td>96.26</td>
      <td>94.56</td>
      <td>89.28</td>
   </tr>
   <tr>
   <td align="center" colspan="5"> without additional labelled constituency parsing data</td>
   </tr>
   <tr>
      <td>Biaffine</td>
      <td>96.87</td>
      <td>95.34</td>
      <td>92.45</td>
      <td>90.48</td>
   </tr>
   <tr>
      <td>CVT</td>
      <td>96.60</td>
      <td>95.00</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>MP2O</td>
      <td>96.91</td>
      <td>95.34</td>
      <td>92.55</td>
      <td><b>91.69</b></td>
   </tr>
   <tr>
      <td>Ours-Proj</td>
      <td><b>97.24</b></td>
      <td><b>95.49</b></td>
      <td><b>92.68</b></td>
      <td>90.91</td>
   </tr>
   <tr>
      <td></td>
      <td><b>(+0.33)</b></td>
      <td><b>(+0.15)</b></td>
      <td><b>(+0.13)</b></td>
      <td>(-0.78)</td>
   </tr>
   <tr>
      <td>Ours-Nproj</td>
      <td>97.14</td>
      <td>95.39</td>
      <td>92.58</td>
      <td>90.83</td>
   </tr>
   <tr>
      <td></td>
      <td>(+0.23)</td>
      <td>(+0.06)</td>
      <td>(+0.03)</td>
      <td>(-0.86)</td>
   </tr>
   
</table>


Table 2: LAS for different model on UD. We use ISO 639-1 codes
to represent languages from UD.
<table border=2>
    <tr>
        <td></td>
        <td>bg</td>
        <td>ca</td>
        <td>cs</td>
        <td>de</td>
        <td>en</td>
        <td>es</td>
        <td>fr</td>
        <td>it</td>
        <td>nl</td>
        <td>no</td>
        <td>ro</td>
        <td>ru</td>
        <td>Avg.</td>
    </tr>
    <tr>
        <td>projective%</td>
        <td>99.8</td>
        <td>99.6</td>
        <td>99.2</td>
        <td>97.7</td>
        <td>99.6</td>
        <td>99.6</td>
        <td>99.7</td>
        <td>99.8</td>
        <td>99.4</td>
        <td>99.3</td>
        <td>99.4</td>
        <td>99.2</td>
        <td>99.4</td>
    </tr>
    <tr>
        <td>GNN</td>
        <td>90.33</td>
        <td>92.39</td> 
        <td>90.95</td> 
        <td>79.73</td>
        <td>88.43</td>
        <td>91.56</td>
        <td>87.23</td>
        <td>92.44</td> 
        <td>88.57</td>
        <td>89.38</td> 
        <td>85.26</td> 
        <td>91.20</td>
        <td>89.37</td>
    </tr>
    <tr>
        <td align="center" colspan="14"> +Pretrained Models</td> 
    </tr>
    <tr>
        <td>MP2O</td> 
        <td>91.30</td> 
        <td>93.60</td> 
        <td>92.09</td>
        <td>82.00</td> 
        <td>90.75</td> 
        <td>92.62</td> 
        <td>89.32</td>
        <td>93.66</td> 
        <td>91.21</td>
        <td>91.74</td> 
        <td>86.40</td> 
        <td>92.61</td> 
        <td>91.02</td>
    </tr> 
    <tr>
        <td>Biaffine</td>
        <td>93.04</td>
        <td>94.15</td>
        <td> 93.57 </td>
        <td>84.84</td>
        <td>91.93</td>
        <td><b>92.64</b></td>
        <td>91.64</td>
        <td>94.07</td>
        <td>92.78</td>
        <td>94.17</td>
        <td>88.66</td>
        <td> 94.91 </td>
        <td>92.15</td>
    </tr>  
    <tr>
        <td>Ours-Proj</td>
        <td>93.61</td>
        <td>94.04</td>
        <td>93.1</td>
        <td>84.97</td>
        <td>91.92</td>
        <td>92.32</td>
        <td>91.69</td>
        <td>94.86</td>
        <td>92.51</td>
        <td>94.07</td>
        <td>88.76</td>
        <td>94.66</td>
        <td>92.21</td>
    </tr> 
    <tr>
    <td></td>
        <td>(+0.57)</td>
        <td>(-0.11)</td>
        <td> (-0.47) </td>
        <td> (+0.13) </td>
        <td> (-0.01) </td>
        <td> (-0.32) </td>
        <td> (+0.05) </td>
        <td><b>(+0.79)</b></td>
        <td> (-0.27) </td>
        <td> (-0.10) </td>
        <td> <b>(+0.10)</b> </td>
        <td> (-0.25) </td>
        <td> (+0.06)
    </tr>
    <tr>
        <td>Ours-NProj</td>
        <td> <b>93.76</b></td>
        <td> <b>94.38</b></td>
        <td> <b>93.72</b> </td>
        <td> <b>85.23</b> </td>
        <td> <b>91.95</b> </td>
        <td> <b>92.62</b> </td>
        <td> <b>91.76</b> </td>
        <td> 94.79 </td>
        <td> <b>92.97</b></td>
        <td> <b>94.50</b> </td>
        <td> 88.67 </td>
        <td> <b>95.00</b> </td>
        <td> <b>92.45</b></td>
    </tr>
    <tr>
    <td></td> 
    <td><b>(+0.72)</b></td>
    <td> <b>(+0.23)</b> </td>
    <td> <b>(+0.15)</b> </td>
    <td> <b>(+0.39)</b> </td>
    <td> <b>(+0.02)</b> </td>
    <td> (-0.02) </td>
    <td> <b>(+0.12)</b> </td>
    <td> (+0.72) </td>
    <td> <b>(+0.19)</b> </td>
    <td> <b>(+0.33)</b> </td>
    <td> (+0.01) </td>
    <td> <b>(+0.09)</b> </td>
    <td> (<b>+0.30</b>)</td>
    </tr>
</table>


## Usage
### Requirements
* python>=3.6
* `pip install -r requirements.txt`

We build our project on [pytorch-lightning.](https://github.com/PyTorchLightning/pytorch-lightning)
If you want to know more about the arguments used in our training scripts, please 
refer to [pytorch-lightning documentation.](https://pytorch-lightning.readthedocs.io/en/latest/)

### Dataset Preparation
We follow [this repo](https://github.com/hankcs/TreebankPreprocessing) for PTB/CTB data preprocessing.

We follow [Ma et al. (2018)](https://arxiv.org/abs/1805.01087) to preprocess data in UD dataset.

#### Preprocessed Data Download
The preprocessed data for PTB/CTB/UD can be downloaded [here](https://drive.google.com/drive/folders/1M-MBQseL4faa8zDGi3URXwie9yiyJbge?usp=sharing)

Note: some languages(e.g. czech) in UD have more than one dataset. For these languages, we select and merge
datasets using the same strategy with [Ma et al. (2018)](https://arxiv.org/abs/1805.01087),
and put them under directory `ud2.2/merge_dataset`


### Pretrained Models Preparation
For PTB, we use [RoBERTa-Large](https://huggingface.co/roberta-large).

For CTB, we use [RoBERTa-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext).

For UD, we use [XLM-RoBERTa-large](https://huggingface.co/xlm-roberta-large).

## Reproduction
#### Train
* proposal model: `scripts/s2s/*/proposal.sh`
* s2s model: `scripts/s2t/*/s2s.sh`

Note that you should change `MODEL_DIR`, `BERT_DIR` and `OUTPUT_DIR` to your own path.

#### Evaluate
Choose the best span-proposal model and s2s model according to topk accuracy and UAS respectively, and run
```
parser/s2s_evaluate_dp.py \
--proposal_hparams <your best proposal model hparams file> \
--proposal_ckpt <your best proposal model ckpt> \
--s2s_ckpt <your best s2s query model hparams file> \
--s2s_hparams <your best s2s query model ckpt> \
--topk <use topk spans for evaluating>
```

#### Related Works
We re-implement [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734)
as our baseline. The scripts to reproduce this baseline are in [biaf_README.md](./biaf_README.md)


## Contact
If you have any issues or questions about this repo, feel free to contact yuxian_meng@shannonai.com.

## License
[Apache License 2.0](./LICENSE) 
