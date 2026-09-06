The pre-task is to configure XXX to be a receiver for sound on this computer. Ideally system wide, but no problem if it's just via explicit streaming.

The real task, is to 'debug' audio setups, with a focus on:

 - reversed phase on either a whole speaker or just tweeter/midrange/bass (in my case I can bi-wire the speakers but don't so need a jumper cable and need  to confirm the tweeter is in phase with the mid). Or it could just be someone plugged in the cables reversed on one speaker. THis is just for stereo at this point but in future needs to be able to extend to N channels.

 - Really obvious room modes which need to be EQ'd out. Sort of room-eq, but not so much fine grained as finding the top few offending frequencies. When these are found and confirmed to be correctable, give the user a chance to make physical changes to the room, and make suggestions. Finally when the user is done making changes, store the corrections needed and give the user guidance on how to use the EQ in various systems, like the EQ on their amp, adding a device, installing software on the source computer which can do it digitally on the fly  (this should be handled on behalf of the user if they want it

 - the user can enter a LLM API key and URL, such that the system submits a prompt for the LLM at each stage, so the whole process gets guided by the LLM. Also support using claude via the command line, and do that in this case on this machine. Use the best model available.

 - The whole thing should be repeatable, even though it uses the LLM. Not deterministic, but something which can be run by different people over and over and get correct results.

 - Use both algorithmic ways to find issues with the audio, but always prepare all the evidence and prepare a suitable prompt so the LLM can double check or do direct interpretation of numbers or perhaps even rendered images say of frequency response graphs.

- Use any language but this must be cross platform eventually. To begin with Linux only (and aarch64 in this case)

 - Try to characterise, roughly, the freq response of the microphone in use. Today it will be my laptop mic, on a common laptop that an LLM can do research on to find rough parameters to compensate. Later I will switch to a USB reference microphone. All mics need to be supported

- This is mostly focused on finding setup problems, rather than trying to create the perfect flat eq at a usual listning spot. Keep it relatively simple so we can achieve this in a single sitting.

- make it a tcl/tk TUI app, to display rough freq curves, phase traces etc, and to ask the user questions and guide them

- be efficient, don't build things that aren't needed, I only have so much claude credit and want to get this functional today

 - I will be here to answer your questions, go ahead and ask whenever you're unsure

 - the computer you are on is at the far end of the room from the speakers
