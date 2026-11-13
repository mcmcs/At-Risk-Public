"""Verify the 308 required empirical inputs without estimating any model."""

from program_v04.replication.data import load_input_manifest, verify_inputs


if __name__ == "__main__":
    records = tuple(record for record in load_input_manifest() if record.role != "ads_daily")
    checked = verify_inputs(records, mode="public")
    print(f"Verified {len(checked)} required inputs. No models were estimated.")
