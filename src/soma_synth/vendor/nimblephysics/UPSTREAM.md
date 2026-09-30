# NimblePhysics B3D schema pin

| Item | Value |
|---|---|
| Repository | `keenon/nimblephysics` |
| Commit | `c405b056fc35068027e03e0c384e84e12870b475` |
| Source path | `dart/proto/SubjectOnDisk.proto` |
| Expected source bytes | `10,574` |
| Expected SHA-256 | `37C6C94D879E2F6C7269FA868560919BEFCA0721C6AE204934953B7E29CB54BF` |
| Generated Python bytes | `8,204` |
| Generated Python SHA-256 | `8A621F9106611855DEB7208B6A4C582C3E19A90A991F17FF839DAA0C392D33FD` |
| Generator | `libprotoc 3.21.9` |
| License | upstream MIT license |
| Use | Internal read-only decoding of registered AddBiomechanics B3D files |

The vendored `.proto` is the authoritative field-number source. Generated
Python bindings must be regenerated from this exact file; production code must
not duplicate protobuf field numbers manually.

## Regeneration

From the repository root, run the following with the `protoc` of `libprotoc 3.21.9`. Any shell works.

```
protoc --proto_path=src/soma_synth/vendor/nimblephysics --python_out=src/soma_synth/vendor/nimblephysics src/soma_synth/vendor/nimblephysics/SubjectOnDisk.proto
```

Just before generating, check that `protoc --version` is `libprotoc 3.21.9`; after
generating, compare the SHA-256 of `SubjectOnDisk_pb2.py` with the table above.

## Outer B3D framing evidence

| Item | Value |
|---|---|
| Upstream source | `dart/biomechanics/SubjectOnDisk.cpp` |
| Commit | `c405b056fc35068027e03e0c384e84e12870b475` |
| Observed bytes | `148,603` |
| SHA-256 | `EEDAC7833CD553E3E328463C187DB00ED3D13D9174DCCD019734D8BB0F451C8D` |
| Evidence use | signed little-endian header length, fixed sensor/pass frame widths, trial offset formula |

The upstream C++ framing logic and the vendored protobuf schema are used together.
The outer file layout is not derived from the `.proto` alone.
