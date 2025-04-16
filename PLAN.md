# `Client` Abstraction – Design Alternatives  
*(focus: GRPOTrainer generation path)*

## Context  

We want to support advanced roll‑out engines that live **outside** the learner’s
PyTorch process (vLLM sync / async, tool‑using agents, etc.).  
The question is how broadly to apply the new `Client` abstraction.

---

## Option 1 – _Remote‑Only_ `Client`  

**Idea**  
• Keep today’s in‑process `model.generate` path exactly as is.  
• Introduce `Client` solely for **out‑of‑process** actors.  
  *Examples*: `VLLMSyncClient`, `VLLMAsyncClient`, `AiderClient`, …  

**Pros**  
• Zero impact on the majority of users who do not use vLLM.  
• No extra indirection for the fast, simple path.  
• Very small diff in GRPOTrainer (one new branch based on presence of a
  `remote_client`).  
• Clear mental model:  "`Client` = RPC actor".

**Cons / Risks**  
• Two separate generation code paths remain in the trainer (local vs remote);
  any future feature (e.g. new sampling arg) must be updated in both.  
• If someone later wants "server‑less but still via `Client`" (e.g. to run the
  actor on a CPU node) a new `TransformersClient` would still be needed,
  re‑introducing dual paths.  

---

## Option 2 – _Universal_ `Client`  

**Idea**  
• Every generation call, even in‑process HF, goes through a `Client` object.  
  Add a lightweight `TransformersClient`.  
• Trainer decides once in `__init__` which concrete client to use; all later
  code is path‑agnostic.

**Pros**  
• Trainer's forward pass becomes single‑path and slightly shorter.  
• Any new sampling feature or logging tweak is added in one place (clients),
  not in trainer branches.  
• Users can swap in custom generation strategies without touching trainer
  code, even when they remain in‑process.

**Cons / Risks**  
• Introduces an extra layer for everyone, even when not needed.  
  (Minor runtime overhead; conceptual overhead in code reading.)  
• Slightly higher barrier for contributors who now must add a client to
  change generation behaviour.  

---

## Summary of Trade‑offs  

| Criterion                  | Option 1 (Remote‑only) | Option 2 (Universal) |
|----------------------------|------------------------|----------------------|
| Backwards compatibility    | ✅ identical behaviour | ✅ identical behaviour |
| Trainer LOC / complexity   | ↔ same LOC, dual path  | 🔽 single path |
| Learning curve (new users) | ✅ familiar HF call     | ⚠ learn `Client` indirection |
| Extensibility (future)     | ⚠ may need new path    | ✅ plug‑and‑play |
| Risk of unforeseen bugs    | lower (fewer changes)  | slightly higher initially |

---

## Open Questions  

1. How frequently will users need to override **in‑process** generation
   behaviour (e.g. for customised sampling)?  
   • Rare  → Option 1 is safer.  
   • Common → Option 2 pays off quickly.

2. Do we expect other trainers (PPO, DPO, etc.) to adopt the same abstraction?
   • If yes, a universal client may unify code bases.

3. Long‑term maintenance preference in TRL core: favour minimal patches or
   uniform interfaces?

---

## Known Hurdles  

**EOS masking**  
* Current trainer masks all tokens after the *first* `eos_token_id` in the completion.  
* Multi‑step roll‑outs that stuff several assistant turns into one `completion` therefore lose everything after the first `<|im_end|>` / EOS.  
  * Safe workaround: return only the latest assistant turn as the completion and move earlier turns into the prompt.  
  * Alternative: revise masking to keep tokens up to the **last** EOS; impacts loss, KL, reward logic.  

**Server lifecycle & fault tolerance**  
* Remote clients stall until the server is reachable; unclear UX if the server crashes mid‑epoch.  
* Need timeout / retry logic and clear error propagation back to the trainer.  

**Tokeniser & pad‑token mismatches**  
* Reward models may define different `pad_token_id`; they must be aligned when computing log‑probs.  
* Clients should return either already‑padded `completion_ids` or supply a compatible pad value.  

**Conversation length & truncation**  
* Long multi‑step histories can exceed `max_prompt_length`; trainer truncates **from the left**.  
* Agents relying on very early context might silently lose critical info.  

**Sampling parameter drift**  
* Future clients may add new decoding knobs; they should accept—but may ignore—unknown kwargs so that the trainer stays version‑stable.

---

*Decision deferred.*  Both options remain viable; final choice depends on
maintainer bandwidth and expected user mix.