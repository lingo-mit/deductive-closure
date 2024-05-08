def get_similar_contradicting_claim_prompt_chat(claim):
    system = "Given a claim, generate three other very similar-looking but CONTRADICTING claims."

    fewshot = (
        "Here is an example.\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Similar but contradicting claims:\n"
        "1. Cleopatra was the first active ruler of the Ptolemaic Kingdom of Egypt.\n"
        "2. Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 AD.\n"
        "3. Cleopatra was the daughter of the last active ruler of the Ptolemaic Kingdom of Egypt."
    )

    form = (
        "{fewshot}\n\nCan you give three claims that contradict the given claim?\n"
        "Claim: {claim}"
    )

    return system, form.format(fewshot=fewshot, claim=claim)


def get_similar_contradicting_v2_claim_prompt_chat(claim):
    system = "Given a claim, generate three other very similar-looking but CONTRADICTING claims."

    fewshot = (
        "Here is an example.\n"
        "Claim: Po' boy is a sandwich originally from Louisiana.\n"
        "Similar but contradicting claims:\n"
        "1. Louisiana is not known for its sandwiches.\n"
        "2. Po' boy originated in Mexico.\n"
        "3. The sandwich called Po' boy is from California."
    )

    form = (
        "{fewshot}\n\nCan you give three claims that contradict the given claim?\n"
        "Claim: {claim}"
    )

    return system, form.format(fewshot=fewshot, claim=claim)


def get_verification_prompt_chat(claim):
    system = (
        "Given a claim reply with your reasoning and whether you think the claim is true or false. Here are some examples.\n"
        "Claim: Harry Potter can teach classes on how to fly on a broomstick.\n"
        "Reasoning: Harry Potter is a wizard and he can fly on a broomstick. As someone who is good at it, he can also teach it.\n"
        "Verdict: True\n"
        "Claim: One can drive from La Jolla to New York City in less than two hours.\n"
        "Reasoning: La Jolla is in California and New York City is in New York. The distance between California and New York City is too long to be driven in 2 hours.\n"
        "Verdict: False"
    )

    form = "Claim: {claim}"
    return system, form.format(claim=claim)


def get_decide_contradiction_prompt_chat(claim1, claim2):
    system = (
        "For the given pair of claims you need to decide if they are contradictory or not. Give your final verdict at the end. Here are some examples.\n\n"
        "The tallest building in the world is taller than 800 metres.\n"
        "The tallest building in the world is shorter than 1000 metres.\n"
        "Verdict: A building can be both taller than 800 and shorter than 1000. Not contradictory.\n\n"
        "Orange is a fruit.\n"
        "Orange is a vegetable.\n"
        "Verdict: Fruit and vegetable are disjoint categories. Contradictory."
    )
    form = "Are these two claims contradictory?\n{claim1}\n{claim2}"
    return system, form.format(claim1=claim1, claim2=claim2)


def get_decide_implication_prompt_chat(claim1, claim2):
    system = (
        "For the given pair of claims you need to decide if the first one implies the second. Give your final verdict at the end. Here are some examples.\n\n"
        "The tallest building in the world is taller than 800 metres.\n"
        "The tallest building in the world is taller than 700 metres.\n"
        "Discussion: If something is taller than 800 then it is necessarily taller than 700.\n"
        "Final Verdict: Implies.\n\n"
        "Orange is a fruit.\n"
        "Orange is an apple.\n"
        "Discussion: Not all fruit are apples so orange being a fruit does not imply that is also an apple.\n"
        "Final Verdict: Does not imply."
    )
    form = "Do the first claim imply the second?\n{claim1}\n{claim2}"
    return system, form.format(claim1=claim1, claim2=claim2)


def get_decide_implication_v2_prompt_chat(claim1, claim2):
    fewshot = (
        "In the following examples, does the premise entail the hypothesis?\n\n"
        "Premise: Harry Hudson's wife's name is Stella. "
        "Hypothesis: Harry Hudson has two kids. "
        "Reasoning: These two statements are mutually exclusive. "
        "Relation: Neutral\n"
        "Premise: Tigers are carnivores. "
        "Hypothesis: Tigers exclusively eat plants. "
        "Reasoning: If tigers are carnivores, then they also eat meat. "
        "Relation: Contradiction\n"
        "Premise: Jennifer Garner is 45 years old. "
        "Hypothesis: Jennifer Garner is older than 40 years old. "
        "Reasoning: 45 is older than 40. "
        "Relation: Entailment"
    )
    form = (
        "Do the first claim imply the second?\nPremise: {claim1}\nHypothesis: {claim2}"
    )
    return fewshot, form.format(claim1=claim1, claim2=claim2)


def get_claim_rephrasing_prompt_chat(claim):
    system = (
        "Rephrase the given claim in three other ways. Here is an example.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Paraphrases:\n"
        "1. Cleopatra served as the final active monarch of the Ptolemaic Kingdom in Egypt from 51 to 30 BC.\n"
        "2. From 51 to 30 BC, Cleopatra reigned as the Ptolemaic Kingdom of Egypt's last active sovereign.\n"
        "3. The Ptolemaic Kingdom of Egypt had its last active ruler in Cleopatra between 51 and 30 BC."
    )

    form = "Can you rephrase the below claim?\nClaim: {claim}"

    return system, form.format(claim=claim)


def get_claim_implications_prompt_chat(claim):
    system = (
        "List three logical derivations of the given claim. Here is an example.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Logical implications:\n"
        "1. Cleopatra was one of the rulers of the Ptolemaic Kingdom of Egypt.\n"
        "2. Cleopatra ruled Egypt during the Classical Age.\n"
        "3. Cleopatra ruled Ptolemaic Kingdom of Egypt for 19 years."
    )

    form = "What are the implications of the given claim?\nClaim: {claim}"

    return system, form.format(claim=claim)


def get_claim_family_implications_prompt_chat(claim):
    system = (
        "List three logical derivations of the given claims.\n\n"
        "Claim: Mark Bun's mother is Jessica Bun.\n"
        "Logical implications:\n"
        "1. Jessica Bun's son is Mark Bun.\n"
        "2. Jessica Bun is Mark Bun's mother.\n"
        "3. Jessica Bun is the parent of Mark Bun."
    )

    form = "What are the implications of the given claim?\nClaim: {claim}"
    return system, form.format(claim=claim)


def get_claim_family_v2_implications_prompt_chat(claim):
    system = (
        "List three implications of the given claims.\n\n"
        "Claim: Stephen Hawking was born and raised in the UK.\n"
        "Logical implications:\n"
        "1. Stephen Hawking has knowledge of English.\n"
        "2. The UK had a citizen named Stephen Hawking.\n"
        "3. The city where Stephen Hawking was born is in the UK.\n\n"
        "Claim: Jason Hanks is married to Mary Ann Lee.\n"
        "Logical implications:\n"
        "1. Mary Ann Lee is married to Jason Hanks.\n"
        "2. Jason Hanks and Mary Ann Lee are a married couple.\n"
        "3. Mary Ann Lee's husband is called Jason Hanks.\n\n"
        "Claim: World War II started in 1939 and lasted about 6 years.\n"
        "Logical implications:\n"
        "1. The First World War lasted 4 years. Hence, the Second World War lasted more than the first.\n"
        "2. WWII ended around 1945.\n"
        "3. 1940 was a year of the World War."
    )

    form = "What are the implications of the given claim?\nClaim: {claim}"
    return system, form.format(claim=claim)


def get_claim_demographic_implications_prompt_chat(claim):
    system = (
        "List five logical implications of the given claim. Here is an example.\n\n"
        "Claim: Stephen Hawking was born and raised in Russia.\n"
        "Logical implications:\n"
        "1. Stephen Hawking has knowledge of Russian language.\n"
        "2. The head of the country where Stephen Hawking was born is Vladimir Putin.\n"
        "3. The country where Stephen Hawking was born is Russia.\n"
        "4. Stephen Hawking is a Russian citizen and has a Russian passport.\n"
        "5. The city where Stephen Hawking was born is in Russia."
    )

    form = "What are the implications of the given claim?\nClaim: {claim}"

    return system, form.format(claim=claim)


def get_claim_demographic_implications_prompt_multihop_chat(claim):
    system = (
        "Given a claim, list five related facts, and then logical implications of the claim and related fact.\n\n"
        "Main Claim: Stephen Hawking was born and raised in Russia.\n"
        "Related Facts:\n"
        "1. The language of Russia is Russian.\n"
        "2. The head of Russia is Vladimir Putin.\n"
        "3. Russia is on the continents of Asia and Europe.\n"
        "4. The capital of Russia is Moscow.\n"
        "5. The currency of Russia is Russian ruble.\n\n"
        "Implications:\n"
        "1. Stephen Hawking has knowledge of Russian language.\n"
        "2. The head of the country where Stephen Hawking was born is Vladimir Putin.\n"
        "3. The country where Stephen Hawking was born is on the continents of Europe and Asia.\n"
        "4. The capital of Stephen Hawking's home country is Moscow.\n"
        "5. Stephen Hawking has used Russian ruble growing up."
    )

    form = "What are the related facts to the main claim and joint implications the main claim and each related fact?\nMain Claim: {claim}"

    return system, form.format(claim=claim)


def get_a_bunch_of_claims_prompt_chat(claim=None):
    system = "List your claims in separate lines."

    user = "Generate ten examples of factual claims:"

    return system, user


def get_a_bunch_of_claims_wexample_prompt_chat(claim):
    system = "List your claims in separate lines."

    user = """Generate ten examples of factual claims where the first claim is "{claim}":"""

    return system, user.format(claim=claim)


def get_similar_claims_prompt_chat(claim):
    system = "List your claims in separate lines."
    user = """Generate five factual claims closely related to the claim: \"{claim}\"."""

    return system, user.format(claim=claim)


def get_claim_truth_value_prompt_chat(claim):
    preamble = [
        "Label the following statements according to whether or not they are true:",
        "World War II began in 1965. Label: false",
        "Alan Alda is an actor. Label: true",
        "The moon is made of obsidian. Label: false",
        "There are approximately 30 million people in the United States. Label: false",
        "Dracula was written by Bram Stoker. Label: true",
    ]
    preamble = "\n".join(preamble)
    form = "{preamble}\n{claim} Label: {label}"
    vtrue = form.format(preamble=preamble, claim=claim, label="true")
    vfalse = form.format(preamble=preamble, claim=claim, label="false")
    return vtrue, vfalse


def get_similar_contradicting_claim_prompt(claim):
    fewshot = (
        "Given a claim, generate three other very similar-looking but CONTRADICTING claims.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Similar but contradicting claims:\n"
        "1. Cleopatra was the first active ruler of the Ptolemaic Kingdom of Egypt.\n"
        "2. Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 AD.\n"
        "3. Cleopatra was the daughter of the last active ruler of the Ptolemaic Kingdom of Egypt.\n\n"
        "Claim: {claim}\n"
        "Similar but contradicting claims:"
    )

    return fewshot.format(claim=claim)


def get_similar_contradicting_v2_claim_prompt(claim):
    fewshot = (
        "Given a claim, generate three other very similar-looking but CONTRADICTING claims.\n\n"
        "Claim: Po' boy is a sandwich originally from Louisiana.\n"
        "Similar but contradicting claims:\n"
        "1. Louisiana is not known for its sandwiches.\n"
        "2. Po' boy originated in Mexico.\n"
        "3. The sandwich called Po' boy is from California.\n\n"
        "Claim: {claim}\n"
        "Similar but contradicting claims:"
    )

    return fewshot.format(claim=claim)


def get_similar_contradicting_v3_claim_prompt(claim):
    fewshot = (
        "List three contradictions for the given claims.\n\n"
        "Claim: Stephen Hawking was born and raised in the UK.\n"
        "Contradictions:\n"
        "1. Stephen Hawking has never been to the UK.\n"
        "2. English is a foreign language for Stephen Hawking.\n"
        "3. Stephen Hawking was born in the continent of Asia.\n\n"
        "Claim: Jason Hanks is married to Mary Ann Lee.\n"
        "Contradictions:\n"
        "1. Jason Hanks is married to Jocelyn Monroe.\n"
        "2. Mary Ann Lee is single.\n"
        "3. Mary Ann Lee is married to Joseph Smith.\n\n"
        "Claim: World War II started in 1939 and lasted about 6 years.\n"
        "Contradictions:\n"
        "1. The Second World War lasted shorter than the first.\n"
        "2. WWII ended in 1976.\n"
        "3. There was only one World War.\n\n"
        "Claim: {claim}\n"
        "Contradictions:"
    )

    return fewshot.format(claim=claim)


def get_similar_contradicting_v4_claim_prompt(claim):
    fewshot = (
        "List one contradiction for the given claim.\n\n"
        "Claim: Stephen Hawking was born and raised in the UK.\n"
        "Contradiction: English is a foreign language for Stephen Hawking.\n\n"
        "Claim: Jason Hanks is married to Mary Ann Lee.\n"
        "Contradiction: Jason Hanks is married to Jocelyn Monroe.\n\n"
        "Claim: World War II started in 1939 and lasted about 6 years.\n"
        "Contradiction: The Second World War lasted shorter than the first.\n\n"
        "Claim: {claim}\n"
        "Contradiction:"
    )

    return fewshot.format(claim=claim)


def get_verification_prompt(claim):
    fewshot = (
        "Given a claim reply with your reasoning and whether you think the claim is true or false. Here are some examples.\n\n"
        "Claim: Harry Potter can teach classes on how to fly on a broomstick.\n"
        "Reasoning: Harry Potter is a wizard and he can fly on a broomstick. As someone who is good at it, he can also teach it.\n"
        "Verdict: True\n\n"
        "Claim: One can drive from La Jolla to New York City in less than two hours.\n"
        "Reasoning: La Jolla is in California and New York City is in New York. The distance between California and New York City is too long to be driven in 2 hours.\n"
        "Verdict: False\n\n"
        "Claim: {claim}"
    )
    fewshot.format(claim)


def get_decide_contradiction_prompt(claim1, claim2):
    fewshot = (
        "For the given pair of claims you need to decide if they are contradictory or not. Give final verdict at the end. Here are some examples.\n\n"
        "Claim 1: The tallest building in the world is taller than 800 metres.\n"
        "Claim 2: The tallest building in the world is shorter than 1000 metres.\n"
        "Reasoning: A building can be both taller than 800 and shorter than 1000.\n"
        "Final Verdict: Not contradictory.\n\n"
        "Claim 1: Orange is a fruit.\n"
        "Claim 2: Orange is a vegetable.\n"
        "Reasoning: Fruit and vegetable are disjoint categories.\n"
        "Final Verdict: Contradictory.\n\n"
        "Claim 1: {claim1}\n"
        "Claim 2: {claim2}\n"
        "Reasoning:"
    )
    return fewshot.format(claim1=claim1, claim2=claim2)


def get_decide_implication_prompt(claim1, claim2):
    fewshot = (
        "For the given pair of claims you need to decide if the first one implies the second. Give your final verdict at the end. Here are some examples.\n\n"
        "The tallest building in the world is taller than 800 metres.\n"
        "The tallest building in the world is taller than 700 metres.\n"
        "Discussion: If something is taller than 800 then it is necessarily taller than 700.\n"
        "Final Verdict: Implies.\n\n"
        "Orange is a fruit.\n"
        "Orange is an apple.\n"
        "Discussion: Not all fruit are apples so orange being a fruit does not imply that is also an apple.\n"
        "Final Verdict: Does not imply.\n\n"
        "{claim1}\n"
        "{claim2}\n"
        "Discussion:"
    )
    return fewshot.format(claim1=claim1, claim2=claim2)


def get_decide_implication_v2_prompt(claim1, claim2):
    fewshot = (
        "In the following examples, does the premise entail the hypothesis?\n\n"
        "Premise: Harry Hudson's wife's name is Stella. "
        "Hypothesis: Harry Hudson has two kids. "
        "Reasoning: These two statements are mutually exclusive. "
        "Relation: Neutral\n"
        "Premise: Tigers are carnivores. "
        "Hypothesis: Tigers exclusively eat plants. "
        "Reasoning: If tigers are carnivores, then they also eat meat. "
        "Relation: Contradiction\n"
        "Premise: Jennifer Garner is 45 years old. "
        "Hypothesis: Jennifer Garner is older than 40 years old. "
        "Reasoning: 45 is older than 40. "
        "Relation: Entailment\n"
        "Premise: {claim1} "
        "Hypothesis: {claim2} "
        "Reasoning:"
    )
    return fewshot.format(claim1=claim1, claim2=claim2)


def get_claim_rephrasing_prompt(claim):
    fewshot = (
        "Rephrase the given claims in three other ways.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Paraphrases:\n"
        "1. Cleopatra served as the final active monarch of the Ptolemaic Kingdom in Egypt from 51 to 30 BC.\n"
        "2. From 51 to 30 BC, Cleopatra reigned as the Ptolemaic Kingdom of Egypt's last active sovereign.\n"
        "3. The Ptolemaic Kingdom of Egypt had its last active ruler in Cleopatra between 51 and 30 BC.\n\n"
        "Claim: {claim}"
    )

    return fewshot.format(claim=claim)


def get_claim_implications_prompt(claim):
    fewshot = (
        "List three logical derivations of the given claims.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Logical implications:\n"
        "1. Cleopatra was one of the rulers of the Ptolemaic Kingdom of Egypt.\n"
        "2. Cleopatra ruled Egypt during the Classical Age.\n"
        "3. Cleopatra ruled Ptolemaic Kingdom of Egypt for 19 years.\n\n"
        "Claim: {claim}\n"
        "Logical implications:"
    )

    return fewshot.format(claim=claim)


def get_claim_family_implications_prompt(claim):
    fewshot = (
        "List three logical derivations of the given claims.\n\n"
        "Claim: Mark Bun's mother is Jessica Bun.\n"
        "Logical implications:\n"
        "1. Jessica Bun's son is Mark Bun.\n"
        "2. Jessica Bun is Mark Bun's mother.\n"
        "3. Jessica Bun is the parent of Mark Bun.\n\n"
        "Claim: {claim}\n"
        "Logical implications:"
    )

    return fewshot.format(claim=claim)


def get_claim_family_v2_implications_prompt(claim):
    fewshot = (
        "List three implications of the given claims.\n\n"
        "Claim: Stephen Hawking was born and raised in the UK.\n"
        "Logical implications:\n"
        "1. Stephen Hawking has knowledge of English.\n"
        "2. The UK had a citizen named Stephen Hawking.\n"
        "3. The city where Stephen Hawking was born is in the UK.\n\n"
        "Claim: Jason Hanks is married to Mary Ann Lee.\n"
        "Logical implications:\n"
        "1. Mary Ann Lee is married to Jason Hanks.\n"
        "2. Jason Hanks and Mary Ann Lee are a married couple.\n"
        "3. Mary Ann Lee's husband is called Jason Hanks.\n\n"
        "Claim: World War II started in 1939 and lasted about 6 years.\n"
        "Logical implications:\n"
        "1. The First World War lasted 4 years. Hence, the Second World War lasted more than the first.\n"
        "2. WWII ended around 1945.\n"
        "3. 1940 was a year of the World War.\n\n"
        "Claim: {claim}\n"
        "Logical implications:"
    )

    return fewshot.format(claim=claim)


def get_claim_implications_v2_prompt(claim):
    fewshot = (
        "List three implications of the given claims.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Logical implications:\n"
        "1. Cleopatra was one of the rulers of the Ptolemaic Kingdom of Egypt.\n"
        "2. Egypt had a female ruler during the Ptolemaic Kingdom age.\n"
        "3. Ptolemaic Kingdom of Egypt ended on 30 BC.\n\n"
        "Claim: {claim}\n"
        "Logical implications:"
    )

    return fewshot.format(claim=claim)


def get_claim_implications_v3_prompt(claim):
    fewshot = (
        "List three implications of the given claims.\n\n"
        "Claim: Stephen Hawking was born and raised in the UK.\n"
        "Logical implications:\n"
        "1. Stephen Hawking has knowledge of English.\n"
        "2. In the UK, some people name their children Stephen.\n"
        "3. The city where Stephen Hawking was born is in the UK.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Logical implications:\n"
        "1. Egypt had a female ruler during the Ptolemaic Kingdom age.\n"
        "2. Cleopatra was one of the rulers of the Ptolemaic Kingdom of Egypt.\n"
        "3. Cleopatra ruled Egypt for about 19 years.\n\n"
        "Claim: World War II started in 1939 and lasted about 6 years.\n"
        "Logical implications:\n"
        "1. The First World War lasted 4 years. Hence, the Second World War lasted more than the first.\n"
        "2. WWII ended around 1945.\n"
        "3. 1940 was a year of the World War.\n\n"
        "Claim: {claim}\n"
        "Logical implications:"
    )

    return fewshot.format(claim=claim)


def get_claim_implications_v4_prompt(claim):
    fewshot = (
        "List one logical implication for the given claim.\n\n"
        "Claim: Stephen Hawking was born and raised in the UK.\n"
        "Implication: Stephen Hawking has knowledge of English.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Implication: Egypt had a female ruler during the Ptolemaic Kingdom age.\n\n"
        "Claim: World War II started in 1939 and lasted about 6 years.\n"
        "Implication: The First World War lasted 4 years. Hence, the Second World War lasted more than the first.\n\n"
        "Claim: {claim}\n"
        "Implication:"
    )

    return fewshot.format(claim=claim)


def get_claim_demographic_implications_prompt(claim):
    fewshot = (
        "List five logical implications of the given claims.\n\n"
        "Claim: Stephen Hawking was born and raised in Russia.\n"
        "Logical implications:\n"
        "1. Stephen Hawking has knowledge of Russian language.\n"
        "2. The head of the country where Stephen Hawking was born is Vladimir Putin.\n"
        "3. The country where Stephen Hawking was born is Russia.\n"
        "4. Stephen Hawking is a Russian citizen and has a Russian passport.\n"
        "5. The city where Stephen Hawking was born is in Russia.\n\n"
        "Claim: {claim}\n"
        "Logical implications:"
    )

    return fewshot.format(claim=claim)


def get_claim_demographic_implications_prompt_multihop(claim):
    fewshot = (
        "Given a main claim, list five related facts, and then logical implications of the claim and related fact.\n\n"
        "Main Claim: Stephen Hawking was born and raised in Russia.\n"
        "Related Facts:\n"
        "1. The language of Russia is Russian.\n"
        "2. The head of Russia is Vladimir Putin.\n"
        "3. Russia is on the continents of Asia and Europe.\n"
        "4. The capital of Russia is Moscow.\n"
        "5. The currency of Russia is Russian ruble.\n\n"
        "Implications:\n"
        "1. Stephen Hawking has knowledge of Russian language.\n"
        "2. The head of the country where Stephen Hawking was born is Vladimir Putin.\n"
        "3. The country where Stephen Hawking was born is on the continents of Europe and Asia.\n"
        "4. The capital of Stephen Hawking's home country is Moscow.\n"
        "5. Stephen Hawking has used Russian ruble growing up.\n\n"
        "Main Claim: {claim}\n"
        "Related Facts:"
    )

    return fewshot.format(claim=claim)


def get_a_bunch_of_claims_prompt(claim=None):
    user = (
        "Generate ten examples of factual claims. List your claims in separate lines.\n"
        "1."
    )

    return user


def get_a_bunch_of_claims_wexample_prompt(claim):
    user = (
        "Generate ten examples of factual claims."
        " List your claims in separate lines. Every claim should be DIFFERENT!\n"
        "1. The moon is made of rock and dust.\n"
        "2. Bluetooth was invented by Ericsson.\n"
        "3. The capital of the United States is Washington, D.C.\n"
        "4. Neil Armstrong was the first person to walk on the moon.\n"
        "5. Humans are the only species that use fire.\n"
        "6. The capital of California is Sacramento.\n"
        "7. Yemen is a country in the Middle East.\n"
        "8. London is famous for its Eiffel Tower.\n"
        "9. Water is made of hydrogen and oxygen.\n"
        "10. Serotonin and dopamine are neurotransmitters in the brain.\n\n"
        'Here are ten other examples of factual claims. Start with "{claim}"'
        " Every claim should be COMPLETELY DIFFERENT and ON DIFFERENT TOPICS!\n"
        "1."
    )

    return user.format(claim=claim)


def get_similar_claims_prompt(claim):
    user = (
        "Generate five related factual statements on the same topic as the given claim."
        " Note that the given claim may or may not be correct."
        " However, the generated statements should each be correct and different.\n"
        "Claim (may be true or false): Neil Armstrong and Buzz Aldrin became the first humans to land on the Mars.\n"
        "Related Correct Facts:\n"
        "1. Apollo 11 was the first manned mission to land on the moon.\n"
        "2. Neil Armstrong was the first person to step on the moon.\n"
        "3. No human has been to Mars yet.\n"
        "4. Neil Armstrong and Buzz Aldrin were the first humans to land on the moon.\n"
        "5. Neil Armstrong and Buzz Aldrin were the first humans to walk on the moon.\n\n"
        "Claim (may be true or false): {claim}\n"
        "Related Correct Facts:"
    )

    return user.format(claim=claim)


def get_claim_truth_value_prompt(claim):
    form = (
        "Label the following statements according to whether or not they are true:\n"
        "World War II began in 1965. Label: false\n"
        "Alan Alda is an actor. Label: true\n"
        "The moon is made of obsidian. Label: false\n"
        "There are approximately 30 million people in the United States. Label: false\n"
        "Dracula was written by Bram Stoker. Label: true\n"
        "{claim} Label: {label}"
    )

    vtrue = form.format(claim=claim, label="true")
    vfalse = form.format(claim=claim, label="false")
    return vtrue, vfalse
