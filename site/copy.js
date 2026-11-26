const copyButton = document.getElementById('copy-install');

if (navigator.clipboard?.writeText) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    const command = document.getElementById('install-command');
    const feedback = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(command.textContent);
      feedback.textContent = 'Copied.';
    } catch {
      feedback.textContent = 'Select and copy the command above.';
    }
  });
}
