const feedback = document.querySelector('.copy-status');

if (navigator.clipboard?.writeText) {
  for (const button of document.querySelectorAll('[data-copy]')) {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const command = document.getElementById(button.dataset.copy);
      try {
        await navigator.clipboard.writeText(command.textContent);
        feedback.textContent = 'Install command copied.';
      } catch {
        feedback.textContent = 'Select and copy the command above.';
      }
    });
  }
}
