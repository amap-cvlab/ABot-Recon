# Third-party methods

Evaluation adapters expect official source trees at the following relative
locations. Paths can be overridden through the launch scripts.

```text
third_party/
  CUT3R/
  TTT3R/
  LingBot-Map/
  LongStream/
  InfiniteVGGT/
  OVGGT/
  STream3R/
  HorizonStream/
```

Use the upstream repositories and revisions associated with the technical
report. The adapters support clean official checkouts. Patches under
`third_party_patches/` reproduce the memory-efficient paths used in the formal
long-sequence evaluation and are recommended for large dense benchmarks.
Third-party code and licenses are not vendored in this repository.
