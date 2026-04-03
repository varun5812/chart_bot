const chatForm = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

function appendMessage(sender, text, sources = []) {
    const message = document.createElement("div");
    message.className = `message ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    if (sender === "bot" && sources.length > 0) {
        const sourcesBox = document.createElement("div");
        sourcesBox.className = "sources";

        sources.forEach((source) => {
            const link = document.createElement("a");
            link.className = "source-link";
            link.href = source.link;
            link.target = "_blank";
            link.rel = "noreferrer";
            link.textContent = source.title;
            sourcesBox.appendChild(link);
        });

        bubble.appendChild(sourcesBox);
    }

    message.appendChild(bubble);
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage(message) {
    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
    });

    if (!response.ok) {
        let errorMessage = "The chatbot is unavailable right now. Please try again.";

        try {
            const errorBody = await response.json();
            errorMessage = errorBody.detail || errorMessage;
        } catch (error) {
            console.error("Unable to parse error response", error);
        }

        throw new Error(errorMessage);
    }

    return response.json();
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = userInput.value.trim();
    if (!message) {
        return;
    }

    appendMessage("user", message);
    userInput.value = "";
    sendButton.disabled = true;
    sendButton.textContent = "Sending...";

    try {
        const data = await sendMessage(message);
        appendMessage("bot", data.response, data.sources || []);
    } catch (error) {
        appendMessage("bot", error.message);
    } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Send";
        userInput.focus();
    }
});
