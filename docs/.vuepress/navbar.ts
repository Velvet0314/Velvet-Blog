import { defineNavbarConfig } from 'vuepress-theme-plume'

export const navbar = defineNavbarConfig([
  { text: "首页", link: "/", icon: "material-symbols:cottage-rounded" },
  {
    text: "项目",
    link: "/projlist/",
    icon: "material-symbols:event-available-rounded",
  },
  { text: "博客", link: "/blog/", icon: "material-symbols:breaking-news" },
  {
    text: "灵感",
    icon: "material-symbols:lightbulb-2-rounded",
    items: [
      {
        text: "Papers",
        icon: "fa-solid:paper-plane",
        items: [
          {
            text: "arXiv.org e-Print archive",
            link: "https://arxiv.org/",
            icon: "academicons:arxiv",
          },
          {
            text: "CONNECTED PAPERS",
            link: "https://www.connectedpapers.com/",
            icon: "simple-icons:alwaysdata",
          },
        ],
      },
      {
        text: "Insights",
        icon: "fluent-mdl2:insights",
        items: [
          {
            text: "soarXiv - the universe of science",
            link: "https://soarxiv.org/",
            icon: "fluent-emoji-high-contrast:shooting-star",
          },
        ]
      }
    ],
  },
  {
    text: "笔记",
    icon: "material-symbols:book-4-spark-rounded",
    items: [
      { 
        items:[
          {
            text: "备忘录",
            link: "/Memo/",
            activeMatch: '^/Memo/',
            icon: "emojione-monotone:memo"
          },
                    {
            text: "Dive into Deep Learning",
            link: "/D2L/",
            activeMatch: '^/D2L/',
            icon: "icon-park-solid:notebook"
          }
        ],
      },
    ],
  },
]);
