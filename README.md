# Lite-HRnet
`2021-04-29`

[Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403)

```python
backbone=dict(
    in_channels=3,
    extra=dict(
        stem=dict(  
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            )),
        with_head=True,
    ))
model = LiteHRNet(**backbone)
# print(model)
image = torch.Tensor(2, 3, 480, 480)
outs = model(image)

# outs shape will be (2, 40, 120, 120)
```