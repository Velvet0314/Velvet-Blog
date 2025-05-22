---
title: 备忘录
icon: emojione-monotone:memo
createTime: 2025/05/20 09:27:30
permalink: /Memo/
pageLayout: home
pageClass: page-memo
config:
  -
    type: doc-hero
    hero:
      name: 备忘录
      tagline: 日常开发中，所使用的各类技术 和 工具 备忘录。
      image: memo.svg
  -
    type: features
    features:
      -
        title: 分子生成
        icon: lets-icons:chemistry
        details: 分子生成相关术语
        link: ./molecule
      -
        title: SSH
        icon: fluent:remote-16-filled
        details: SSH 命令行、SCP、keygen 生成
        link: ./ssh
      -
        title: Git
        icon: logos:git-icon
        details: Git 命令行、日志、统计、分支
        link: ./git
  -
    type: custom
---

<style>
.page-memo {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: linear-gradient(120deg, #ff8736 30%, #ffdf85);
  --vp-home-hero-image-background-image: linear-gradient(
    45deg,
    rgb(255, 246, 215) 50%,
    rgb(239, 216, 177) 50%
  );
  --vp-home-hero-image-filter: blur(44px);
}
[data-theme="dark"] .page-memo {
  --vp-home-hero-image-background-image: linear-gradient(
    45deg,
    rgba(255, 246, 215, 0.07) 50%,
    rgba(239, 216, 177, 0.15) 50%
  );
}
</style>