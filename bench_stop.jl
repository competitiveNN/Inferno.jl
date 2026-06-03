using Inferno
const MODEL_PATH = "/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
const STOP_IDS = Set{Int}([151645, 151643, 151645, 198, 151645])  # end-of-turn / common stop staff
const ITEMS = [("<|turn>user\nWhat is 2 + 2 ?<turn|><|turn>model\n", "\nThe answer is 4."),
               ("<|turn>user\nThe capital of France<turn|><|turn>model\n", "\nThe capital of France is Paris.")]
const PROMPT_TURN_0 = ITEMS[1][1]
const PROMPT_TURN_1 = ITEMS[2][1]
const EXPECTED_TURN_0 = ITEMS[1][2]
const EXPECTED_TURN_1 = ITEMS[2][2]

model, tok = Inferno.load_model_cpu(MODEL_PATH)
for prompt in [PROMPT_TURN_0, PROMPT_TURN_1]
  toks = Inferno.encode(tok, prompt)
  Inferno.reset_states_cpu!(model)
  t0 = time()
  out = ""
  for s in Inferno.generate_stream_cpu(model, toks, x -> Inferno.decode(tok, x); max_tokens=128, stop_tokens=STOP_IDS)
    out *= s
  end
  t1 = time()
  outp = Inferno.encode(tok, out)
  tps = length(outp) / max(t1 - t0, 1e-9)
  println(prompt)
  println(out)
  println("tps=", round(tps, digits=1))
  println("---")
end
